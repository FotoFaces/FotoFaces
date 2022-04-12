"""CD Chat server program."""
import logging
import selectors
import socket
import json

from Protocol import FotoFacesProtocol, FotoFacesProtocolBadFormat

logging.basicConfig(filename="api.log", level=logging.DEBUG)

class Api:

    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sel = selectors.DefaultSelector()
        self.sock.bind(('', 8005))
        self.sock.listen(100)
        self.sel.register(self.sock, selectors.EVENT_READ, self.accept)
        print("Starting REST API...")

    def accept(self, sock, mask):
        conn, addr = sock.accept()
        self.sel.register(conn, selectors.EVENT_READ, self.read)

    def read(self, conn, mask):
        data_message = FotoFacesProtocol.recv_msg(conn)                                  

        if not data_message:                                                   
            print('-----Closing Connection to: {}-----'.format(conn))      
            self.sel.unregister(conn)                                         
            conn.close()                                                
            return

        logging.debug("Received: %s", data_message)                             
        data = data_message.json
        if data["command"] == "savePhoto":
            print('Wanting to save photo of student with id {} - photo: {}'.format(data["identification"], data["photo"]))
        elif data["command"] == "getPhoto":
            print('Wanting to get the photo of student with id {}'.format(data["identification"]))

            # database shit and all

            message = FotoFacesProtocol.sendPhoto("photo")
            logging.debug("Sending: ", message) 
            print("Sending: ", message)
            FotoFacesProtocol.send_msg(conn, message)
        else:
            raise FotoFacesProtocolBadFormat(message_encoded)

    def run(self):
        while True:                                                    
            events = self.sel.select()   
            for key, mask in events:
                callback = key.data
                callback(key.fileobj, mask)

    # def broadcast(self, message, channel):                              # envia para os clients todos
    #     for client in self.connecs:                                     # para cada socket
    #         if str(channel) in self.connecs[client][1]:                 # se o channel for igual ao da data a ser enviada enato enviada para essa socket
    #             FotoFacesProtocol.send_msg(client, message)             # send data


if __name__ == "__main__":
    api = Api()
    api.run()