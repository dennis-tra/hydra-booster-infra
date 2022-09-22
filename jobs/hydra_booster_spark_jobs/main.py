import base64
import multihash as mh
import multiaddr as ma
from awsglue import DynamicFrame
from netaddr import IPAddress
from datetime import datetime
from awsglue.transforms import ApplyMapping
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, BooleanType
from pyspark.context import SparkContext
from pyspark import SparkFiles
from awsglue.context import GlueContext
from awsglue.job import Job

# Initialization
spark_context = SparkContext.getOrCreate()
glue_context = GlueContext(spark_context)
spark = glue_context.spark_session
job = Job(glue_context)

# Constants
geo_db_path = 'GeoLite2-Country.mmdb'
spark_context.addFile(geo_db_path)
partition_date = datetime.now().strftime("%Y-%m-%d")
partition_date = "2022-09-20"  # TMP: overwrite

# Load data to work with
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


# Convert base58 encoded PeerID multihash to base64
@F.udf(returnType=StringType())
def peer_id_to_b64(peer_id: str) -> str:
    """
    Takes a peer ID like `QmSKVUFAyCddg2wDUdZVCfvqG5YCwwJTWY1HRmorebXcKG` and transforms it to its base64 representation
    like: `EiA7Irhn7hABRTt+2L/amsK5C8hVAxR42ZWLTTFXjDM+Pw==`
    We do this because multiple encodings could lead to different string representations. The hydra DynamoDB keys
    use base64 encodings, so we convert the PeerIDs here to join the data together.

    However, I just read [here](https://docs.libp2p.io/concepts/peer-id/):
    > PeerIds always use the base58 encoding, with no multibase prefix when encoded into strings.

    :param str peer_id: A base58 encoded PeerID like `QmSKVUFAyCddg2wDUdZVCfvqG5YCwwJTWY1HRmorebXcKG`.
    :return: Base64 encoding of the unterlying multihash information.
    """
    return base64.b64encode(mh.from_b58_string(peer_id)).decode('ascii')


df = df.withColumn("peer_id_b64", peer_id_to_b64("peer_id"))

# Only use subset of columns to work with and unnest/explode the maddrs array to separate columns
df = df.select("date", "peer_id_b64", F.explode("maddrs").alias("maddr"))


# Extract IP address from Multiaddresses
@F.udf(returnType=StringType())
def ip_addr_from_maddr(maddr_str: str) -> str:
    """
    Takes a multi address string like `/ip4/123.345.456.567/tcp/1234` and extracts the IPv4 or IPv6 IP address.

    :param maddr_str: The Multiaddress string like `/ip4/123.345.456.567/tcp/1234`
    :return: The encoded IP address or None if no IPv4 or IPv6 protocol found.
    """
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


df = df.withColumn("ip_address", ip_addr_from_maddr("maddr"))


# Remove all Multiaddresses that are relay addresses.
@F.udf(returnType=BooleanType())
def is_relay(maddr_str: str) -> bool:
    """
    Takes a multi address string like `/ip4/123.345.456.567/tcp/1234/p2p/QmFoo...` and determines if it's a relayed Multiaddress.

    :param maddr_str: The Multiaddress string like `/ip4/123.345.456.567/tcp/1234/p2p/QmFoo...`
    :return: True if it's a relayed Multiaddress and False if it isn't.
    """
    try:
        ma.Multiaddr(maddr_str).value_for_protocol("p2p-circuit")
        return True
    except Exception:
        return False


df = df.where(is_relay('maddr') == False)


# Remove all Multiaddresses that are not publicly reachable.
@F.udf(returnType=BooleanType())
def is_public(ip_address_str: str) -> bool:
    """
    Takes an IP address string like `'123.345.456.567'` and determines if it's publicly reachable.

    :param ip_address_str: The IP address string like `'123.345.456.567'`.
    :return: True if it's a publicly reachable IP address and False if it isn't (e.g. loopback, private, reserved).
    """
    if ip_address_str is None:
        return False

    ip_address = IPAddress(ip_address_str)
    return not (
            ip_address.is_private() or
            ip_address.is_loopback() or
            ip_address.is_reserved() or
            ip_address.is_multicast()
    )


df = df.where(is_public('ip_address'))


# Associate country code with IP addresses
def partition_ip_to_country(partition):
    """
    Function to be passed to `mapPartitions`. This function is called for each partition of the data. We need to do this
    because the GeoIP2 database cannot be pickled and transferred to each partition, hence we need to initialize it
    at each partition (e.g., a User-Defined-Function wouldn't work here).

    :param partition: List of rows in that partition
    """
    import geoip2.database
    with geoip2.database.Reader(SparkFiles.get(geo_db_path)) as reader:
        for row in partition:
            try:
                yield [row.peer_id_b64, reader.country(row.ip_address).country.iso_code]
            except Exception:
                yield [row.peer_id_b64, None]


df = df.rdd.mapPartitions(partition_ip_to_country).toDF(["peer_id_b64", "country"])

# A single peer can have many Multiaddresses all associated with a single country. We only need this mapping once.
# Remove duplicates from the data.
df = df.distinct()

# Convert PySpark DataFrame to AWS Glue DynamicFrame
dyf = DynamicFrame.fromDF(df, glue_context, "peer-store-cleansed")

# Write data to S3
glue_context.write_dynamic_frame.from_options(
    frame=dyf,
    connection_type="s3",
    format="glueparquet",
    connection_options={
        "path": f"s3://dynamodb-exports-686311013710-us-east-2/hydra-test-peerstore-cleansed/date={partition_date}-2/"
    },
    format_options={"compression": "gzip"},
    transformation_ctx="peer_records_cleansed_write_1",
)

# Finalize job
job.commit()
