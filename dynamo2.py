import boto3
import time
import json

table = 'StationDetectTable'
dynamo = boto3.resource('dynamodb')
# for tbl in dynamo.tables.all():
#     print(tbl.name)
dynamo_table = dynamo.Table(table)

with open('out/log.json', mode='r') as f:
    logs = json.load(f)
if len(logs)==0:
    print('No files')
else:
    for log in logs:
        dynamo_table.put_item(Item=log)

time.sleep(3)
