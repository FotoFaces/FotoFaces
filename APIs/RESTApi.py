from flask import Flask, flash, jsonify, request
from flask_restful import Resource, Api
import sqlite3 as sql
import logging
import json

# logger
logging.basicConfig(filename="api.log", level=logging.DEBUG)


# Flask app and Flask Restful api
app = Flask(__name__)
app.logger.debug("\n\n\n\t\tFlask app start")
api = Api(app)


# mock database created with sqlite3
with sql.connect('mydb') as con:
    cur = con.cursor()
    cur.execute("""CREATE TABLE if not exists User (
        id int primary key,
        email text,
        full_name text,
        password varchar,
        photo blob);""")
    cur.close()


# get all users with all the information
# debug material
@app.route("/")
def index():

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
                cur.execute(f"SELECT photo FROM User WHERE id = \'{identification}\'")
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
        photo = request.form["photo"]

        # database call
        app.logger.debug("-- Begin -- Database call while updating old photo")
        try :
            with sql.connect('mydb') as con:
                cur = con.cursor()
                # UPDATE <Table of Users> SET <photo> = <inputed photo> WHERE <identification> = <inputed identification>
                cur.execute(f"UPDATE User SET photo = \'{photo}\' WHERE id = \'{identification}\'")
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

    # PUT method
    # input arguments: identification (id or something similar) -> NMEC if possible
    # updates / inserts a user to the database
    def put(self, identification):
        # curl http://localhost:5000/add_user/<id> -d "email=<email>" -d "name=<name>" -d "password=<password>" -d "photo=<photo>" -X PUT

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
                cur.execute(f"INSERT INTO User VALUES (\'{identification}\', \'{email}\', \'{name}\', \'{password}\', \'{photo}\')")
                con.commit()
                cur.close()
                app.logger.debug(f"Successful query")
        # in case we get an unexpected error
        except KeyError as k:
            app.logger.debug(f"Error: {k}")
            return jsonify({"command": "upload_photo", "id": identification, "photo": photo, "error": k})

        app.logger.debug("-- End -- Database call creating / updating user")

        # return identification and photo
        return jsonify({"command": "upload_photo", "id": identification, "photo": photo})


# add Image class to the url so the methods from Image class can get called
# <int:identification> -> identification for the methods
api.add_resource(Image, '/image/<int:identification>')

# add User class to the url so the methods from User class can get called
# <int:identification> -> identification for the method
api.add_resource(User, '/add_user/<int:identification>')


# main
if __name__ == "__main__":
	app.run()



""" References

- https://flask-restful.readthedocs.io/en/latest/quickstart.html#resourceful-routing
- 

"""