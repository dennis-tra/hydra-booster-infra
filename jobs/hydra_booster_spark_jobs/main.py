import sys
import base64
import multihash as mh
import multiaddr as ma
from awsglue import DynamicFrame
from netaddr import IPAddress
from datetime import datetime
from awsglue.transforms import ApplyMapping
from awsglue.utils import getResolvedOptions
from pyspark.sql.functions import udf, explode, collect_list
from pyspark.context import SparkContext
from pyspark import SparkFiles
from awsglue.context import GlueContext
from awsglue.job import Job

## @params: [JOB_NAME]
args = getResolvedOptions(sys.argv, ['JOB_NAME'])


# Create Spark "User defined function" to map PeerID to its base64 pendant
def peer_id_to_b64(peer_id):
    return base64.b64encode(mh.from_b58_string(peer_id)).decode('ascii')


# Need to do the detour via mapPartitions because of errors around
#   'PicklingError: Could not serialize object'
# for the geoip2 database.
def partition_ip_to_country(partition):
    import geoip2.database
    with geoip2.database.Reader(SparkFiles.get(geo_db_path)) as reader:
        for row in partition:
            try:
                country = reader.country(row.ip_address).country.iso_code
            except Exception:
                country = None
            yield [
                row.date,
                row.peer_id,
                row.peer_id_b64,
                row.maddr,
                row.ip_address,
                row.is_relay,
                row.is_public,
                country,
            ]


# Add IP-Address column
def ip_addr_from_maddr(maddr_str):
    try:
        maddr = ma.Multiaddr(maddr_str)
    except ValueError:
        return None

    try:
        return maddr.value_for_protocol("ip4")
    except Exception:
        pass

    try:
        return maddr.value_for_protocol("ip6")
    except Exception:
        return None


def is_relay(maddr_str):
    try:
        ma.Multiaddr(maddr_str).value_for_protocol("p2p-circuit")
        return True
    except Exception:
        return False


# Create Spark "User defined function" to determine if an address is private
def is_public(ip_address_str):
    if ip_address_str is None:
        return False

    ip_address = IPAddress(ip_address_str)
    return not (
            ip_address.is_private() or
            ip_address.is_loopback() or
            ip_address.is_reserved() or
            ip_address.is_multicast()
    )


# Initialization
spark_context = SparkContext()
glue_context = GlueContext(spark_context)
spark = glue_context.spark_session
geo_db_path = 'GeoLite2-Country.mmdb'
spark_context.addFile(geo_db_path)
job = Job(glue_context)
job.init(args['JOB_NAME'], args)

partition_date = datetime.now().strftime("%Y-%m-%d")
partition_date = "2022-09-20"  # TMP: overwrite

# Get reference to peer store data
dyf = glue_context.create_dynamic_frame.from_catalog(
    database="hydra-test",
    table_name="hydra_test_peerstore",
    push_down_predicate=f"(date == '{partition_date}')",
)

# Flatten data structure
dyf = ApplyMapping.apply(
    frame=dyf,
    mappings=[
        ("AddrInfo.ID", "string", "peer_id", "string"),
        ("AddrInfo.Addrs", "array", "maddrs", "array"),
        ("HeadID", "string", "head_id", "string"),
        ("AgentVersion", "string", "agent_version", "string"),
        ("Protocols", "array", "protocols", "array"),
        ("date", "string", "date", "string")
    ],
    transformation_ctx="peer_records_flatten_1"
)

# Convert to Spark DataFrame
df = dyf.toDF()

# Add base64 peer_id column
df = df.withColumn("peer_id_b64", udf(peer_id_to_b64)("peer_id"))

# Unnest/explode multiaddress column
df = df.select("date", "peer_id", "peer_id_b64", explode("maddrs").alias("maddr"))  # .dropDuplicates(['maddr'])

df = df \
    .withColumn("ip_address", udf(ip_addr_from_maddr)("maddr")) \
    .withColumn("is_relay", udf(is_relay)("maddr")) \
    .withColumn("is_public", udf(is_public)("ip_address"))

df = df.rdd.mapPartitions(partition_ip_to_country).toDF(
    ["date", "peer_id", "peer_id_b64", "maddr", "ip_address", "is_relay", "is_public", "country"])

df = df.where(df["is_relay"] == False).where(df["is_public"] == True)

df = df.groupBy(["date", "peer_id_b64"]).agg(collect_list('country').alias('countries'))

df.show(10)

dyf = DynamicFrame.fromDF(df, glue_context, "peer-store-cleansed")

glue_context.write_dynamic_frame.from_options(
    frame=dyf,
    connection_type="s3",
    format="glueparquet",
    connection_options={
        "path": "s3://dynamodb-exports-686311013710-us-east-2/hydra-test-peerstore-cleansed/2022-09-20/",
        "partitionKeys": ["date"],
    },
    format_options={"compression": "gzip"},
    transformation_ctx="peer_records_cleansed_write_1",
)

job.commit()
