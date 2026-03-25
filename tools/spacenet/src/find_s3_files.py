import boto3
from botocore.config import Config
from botocore import UNSIGNED
#import json
import pandas as pd

BUCKET = "spacenet-dataset"


def s3_client(unsigned=True):
    if unsigned:
        return boto3.client("s3", config=Config(signature_version=UNSIGNED))
    return boto3.client("s3")


def list_objects(bucket, prefix, unsigned=True):
    s3 = s3_client(unsigned=unsigned)
    paginator = s3.get_paginator("list_objects_v2")
    kwargs = {"Bucket": bucket, "Prefix": prefix}

    count = 0
    for page in paginator.paginate(**kwargs):
        obj_list = page.get("Contents", [])
        count += len(obj_list)
        print(f"found {count:,}")
        for obj in obj_list:
            yield {
                "key": obj["Key"],
                "size": obj["Size"],
                "last_modified": obj["LastModified"],
                "etag": obj.get("ETag"),
            }


#prefix = "Hosted-Datasets/CORE3D-Public-Data"
prefix = ""
objs = list(list_objects(BUCKET, prefix, unsigned=True))
df = pd.DataFrame(objs)
df.to_parquet("./artifacts/s3-spacenet-dataset.parquet")

# Convert datetime.datetime to string for json
#for o in objs:
#    o["last_modified"] = str(o["last_modified"])

#with open(f"s3-{BUCKET}.json", "w") as f:
#    json.dump(objs, f, indent=1)
