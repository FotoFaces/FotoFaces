import base64
from flask import Flask, jsonify, request
from flask_restful import Resource, Api
import sqlite3 as sql
import logging
from flask_cors import CORS, cross_origin
import hashlib

# logger
logging.basicConfig(filename="api.log", level=logging.DEBUG)


# Flask app and Flask Restful api
app = Flask(__name__)
app.logger.debug("\n\n\n\t\tFlask app start")
api = Api(app)
cors = CORS(app, resources={r"/*": {"origins": "*"}})


# mock database created with sqlite3
with sql.connect('mydb') as con:
    cur = con.cursor()
    cur.execute("""CREATE TABLE if not exists User (
        id integer primary key autoincrement,
        email text not null unique,
        full_name text,
        password text,
        photo blob);""")
    cur.close()


# get all users with all the information
# debug material
@app.route("/")
@cross_origin()
def index():
    
    ### INSERT DATABASE
    
    # password = hashlib.md5(("adminpg").encode("utf-8", errors='static')).hexdigest()

    # with open("GoncaloOldFoto.jpg", "rb") as f:
    #     ref_bytes = f.read()
    #     goncas = base64.b64encode(ref_bytes).decode("utf8")
        
    # with open("vicente.png", "rb") as f:
    #     ref_bytes = f.read()
    #     vicente = base64.b64encode(ref_bytes).decode("utf8")
        
    # with open("borges.png", "rb") as f:
    #     ref_bytes = f.read()
    #     borges = base64.b64encode(ref_bytes).decode("utf8")
    
    # with open("flips.jpg", "rb") as f:
    #     ref_bytes = f.read()
    #     fip = base64.b64encode(ref_bytes).decode("utf8")
    
    # with sql.connect('mydb') as con:
    #     cur = con.cursor()
    #     cur.execute("INSERT INTO User VALUES(?,?,?,?,?)", (98083, "filipe@ua.pt", "Filipe", password, fip))
    #     cur.execute("INSERT INTO User VALUES(?,?,?,?,?)", (97827, "mini@ua.pt", "Mini", password, "photo"))
    #     cur.execute("INSERT INTO User VALUES(?,?,?,?,?)", (98515, "vicente@ua.pt", "Vicente", password, vicente))
    #     cur.execute("INSERT INTO User VALUES(?,?,?,?,?)", (98155, "borges@ua.pt", "Borges", password, borges))
    #     cur.execute("INSERT INTO User VALUES(?,?,?,?,?)", (98359, "goncas@ua.pt", "Maestro", password, goncas))
    #     con.commit()
    #     cur.close()
    

    # database call
    app.logger.debug("-- Begin -- Database call in index")
    try:
        with sql.connect('mydb') as con:
            cur = con.cursor()
            # SELECT * FROM <Table of Users>
            cur.execute("SELECT * FROM User")
            result = cur.fetchall()
            cur.close()
            # in case we dont get any results
            if result == []:
                # debug
                app.logger.debug(f"{result}")
                return jsonify({"error": "result is empty"})
            app.logger.debug(f"Successful query")
    # in case we get an unexpected error
    except KeyError as k:
        app.logger.debug(f"Error: {k}")
        return jsonify({"error": k})

    app.logger.debug("-- End -- Database call in index")

    # return the results
    return jsonify(result)


# Class generalization for the CRUD methods related to the photo -> database connection
class Image(Resource):

    # GET method
    # input arguments: identification (id or something similar) -> NMEC if possible
    # get the photo in the database for the user with id inputed
    def get(self, identification):
        # curl http://localhost:5000/image/<id> -X GET

        # database call
        app.logger.debug("-- Begin -- Database call while getting the old photo")
        try:
            with sql.connect('mydb') as con:
                cur = con.cursor()
                # SELECT * FROM <Table of Users> WHERE <identification> = <inputed identification>
                cur.execute("SELECT photo FROM User WHERE id = ?", (identification,))
                result = cur.fetchall()
                cur.close()
                # in case we dont get any results
                if result == []:
                    # debug
                    app.logger.debug(f"{result}")
                    return jsonify({"command": "get_photo", "id": identification, "error": "result is empty"})
                app.logger.debug(f"Successful query")
        # in case we get an unexpected error
        except KeyError as k:
            app.logger.debug(f"Error: {k}")
            return jsonify({"command": "get_photo", "id": identification, "error": k})

        app.logger.debug("-- End -- Database call while getting the old photo")

        # extract photo from the list of results
        # result len should be 1 and only 1
        photo = result[0][0]

        # return identification and photo
        return jsonify({"command": "get_photo", "id": identification, "photo": photo})

    # PUT method
    # input arguments: identification (id or something similar) -> NMEC if possible
    # updates the old photo into the new one sent over -> already validated -> simple update
    def put(self, identification):
        # curl http://localhost:5000/image/<id> -d "photo=<photo>" -X PUT

        # get the photo
        photo = request.form["param"]

        # database call
        app.logger.debug("-- Begin -- Database call while updating old photo")
        try :
            with sql.connect('mydb') as con:
                cur = con.cursor()
                # UPDATE <Table of Users> SET <photo> = <inputed photo> WHERE <identification> = <inputed identification>
                cur.execute("UPDATE User SET photo = ? WHERE id = ?", (photo, identification))
                con.commit()
                cur.close()
                app.logger.debug(f"Successful query")
        # in case we get an unexpected error
        except KeyError as k:
            app.logger.debug(f"Error: {k}")
            return jsonify({"command": "upload_photo", "id": identification, "photo": photo, "error": k})

        app.logger.debug("-- End -- Database call while updating old photo")

        # return identification and photo
        return jsonify({"command": "upload_photo", "id": identification, "photo": photo})


# Class generalization for the CRUD methods related to the User -> database connection
# Mock class for testing
class User(Resource):
    
    # GET method
    # input arguments: identification (id or something similar) -> NMEC if possible
    # get the photo in the database for the user with id inputed
    def get(self, param):
        # curl http://localhost:5000/image/<id> -X GET

        email = param

        # database call
        app.logger.debug("-- Begin -- Database call while getting the old photo")
        try:
            with sql.connect('mydb') as con:
                cur = con.cursor()
                # SELECT * FROM <Table of Users> WHERE <identification> = <inputed identification>
                cur.execute(f"SELECT id, photo, password, full_name FROM User WHERE email = \'{email}\'")
                result = cur.fetchall()
                cur.close()
                # in case we dont get any results
                if result == []:
                    # debug
                    app.logger.debug(f"{result}")
                    return jsonify({"command": "get_user", "email": email, "error": "result is empty"})
                app.logger.debug(f"Successful query")
        # in case we get an unexpected error
        except KeyError as k:
            app.logger.debug(f"Error: {k}")
            return jsonify({"command": "get_user", "email": email, "error": k})

        app.logger.debug("-- End -- Database call while getting the old photo")

        # extract photo from the list of results
        # result len should be 1 and only 1
        (identifier, photo, password, name) = result[0]

        # return identification and photo
        return jsonify({"command": "get_user", "email": email, "photo": photo, "password": password, "id": identifier, "name": name})

    # PUT method
    # input arguments: identification (id or something similar) -> NMEC if possible
    # updates / inserts a user to the database
    def put(self, param):
        # curl http://localhost:8393/user/<id> -d "email=<email>" -d "name=<name>" -d "password=<password>" -d "photo=<photo>" -X PUT

        # get variables necessary of a user
        photo = request.form['photo']
        name = request.form['name']
        password = request.form['password']
        email = request.form['email']

        # database call
        app.logger.debug("-- Begin -- Database call creating / updating user")
        try :
            with sql.connect('mydb') as con:
                cur = con.cursor()
                # INSERT INTO <Table of Users> VALUES (<values necessary for the Table of Users>)
                cur.execute("INSERT INTO User (email, full_name, password, photo) VALUES (?,?,?,?)",(email, name, password, photo))
                con.commit()
                cur.close()
                app.logger.debug(f"Successful query")
        # in case we get an unexpected error
        except Exception as k:
            app.logger.debug(f"Error: {k}")
            return jsonify({"command": "upload_photo", "state": "error"})

        app.logger.debug("-- End -- Database call creating / updating user")

        # return identification and photo
        return jsonify({"command": "upload_photo", "state": "success"})


# add Image class to the url so the methods from Image class can get called
# <int:identification> -> identification for the methods
api.add_resource(Image, '/image/<int:identification>')

# add User class to the url so the methods from User class can get called
# <int:identification> -> identification for the method
api.add_resource(User, '/user/<param>')


# main
if __name__ == "__main__":
    # run Flask app
    app.run(debug=True,host="0.0.0.0", port=8393)


    app.logger.info( flask.request.remote_addr)

""" References

- https://flask-restful.readthedocs.io/en/latest/quickstart.html#resourceful-routing
- https://kafka-python.readthedocs.io/en/master/
-

"""
