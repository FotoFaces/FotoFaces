import sqlite3 as sql
import base64
import hashlib
import requests
import json


### INSERT DATABASE
    
# password = hashlib.md5(("adminpg").encode("utf-8", errors='static')).hexdigest()


# with open("images/Pedro.jpg", "rb") as f:
#     ref_bytes = f.read()
#     fip = base64.b64encode(ref_bytes).decode("utf8")

# with sql.connect('../../APIs/mydb') as con:
#     cur = con.cursor()
#     cur.execute("INSERT INTO User VALUES(?,?,?,?,?)", (98083, "filipe@ua.pt", "Filipe", password, fip))
#     con.commit()
#     cur.close()
#     print("Successful insert")

identifier =98083
email ="filipe@ua.pt"
name ="Filipe"
password = hashlib.md5(("adminpg").encode("utf-8", errors='static')).hexdigest()
with open("images/Pedro.jpg", "rb") as f:
    ref_bytes = f.read()
    fip = base64.b64encode(ref_bytes).decode("utf8")
response = requests.put('http://192.168.1.69:8393/user/{identifier}', data= {'email':email,'name' : name, 'password':password})






