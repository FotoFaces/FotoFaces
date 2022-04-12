"""CD Chat client program"""
import logging
import sys
import selectors
import socket
import fcntl
import os
import json

from Protocol import FotoFacesProtocol, FotoFacesProtocolBadFormat

logging.basicConfig(filename=f"{sys.argv[0]}.log", level=logging.DEBUG)
# set sys.stdin non-blocking
#orig_fl = fcntl.fcntl(sys.stdin, fcntl.F_GETFL)
#fcntl.fcntl(sys.stdin, fcntl.F_SETFL, orig_fl | os.O_NONBLOCK)


class Test:

    def __init__(self, identification: int = 1):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.end = 0                                        # variável para saber se devemos terminar o loop ou não
        self.id = identification
        self.selector = selectors.DefaultSelector()
        self.connect()

    def connect(self):
        self.sock.connect(('', 8005))
        self.sock.setblocking(False)
        self.selector.register(self.sock, selectors.EVENT_READ, self.read)

    def read(self, soc, mask):
        data = FotoFacesProtocol.recv_msg(self.sock)
        logging.debug("Received: %s", data.json)            
        print('Received: {}'.format(data))        

    def write_stdin(self, stdin, mask):
        msg = stdin.read()                                  
        if msg.__contains__("exit"):                     
            print("exiting...")
            self.sock.close()                              
            self.selector.unregister(self.sock)            
            self.end = 1                                    
            return

        if msg.__contains__("/save"):                      
            msg_split = msg.split()                         
            if len(msg_split) != 2:                         
                print("Error")
                return
            self.photo = msg_split[1]                               
            message = FotoFacesProtocol.savePhoto(self.id, self.photo)       
        elif msg.__contains__("/get"):      
            message = FotoFacesProtocol.getPhoto(self.id)    
        else:
            print("Nothing happened")
            return

        print(message)
        logging.debug("Sending: %s", message)  
        FotoFacesProtocol.send_msg(self.sock, message)          


    def loop(self):
        orig_fl = fcntl.fcntl(sys.stdin, fcntl.F_GETFL)
        fcntl.fcntl(sys.stdin, fcntl.F_SETFL, orig_fl | os.O_NONBLOCK) 
        self.selector.register(sys.stdin, selectors.EVENT_READ, self.write_stdin)
        sys.stdout.write('Testing \n')
        sys.stdout.flush()                                  
        while True:                                         
            if self.end == 1:                               
                break
            events = self.selector.select()
            for key, mask in events:
                callback = key.data
                callback(key.fileobj, mask)

if __name__ == "__main__":
    test = Test(1)
    test.loop()