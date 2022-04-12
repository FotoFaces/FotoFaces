"""Protocol the Fotofaces project"""
import json
from datetime import datetime
from socket import socket


class Message:
    """Message Type."""
    def __init__(self, command):          
        self.json = {"command": command}      
        if command != "sendPhoto":
            self.json["identification"] = self.id
        if command != "getPhoto":
            self.json["photo"] = self.photo
    
    def __str__(self):                                 
         return json.dumps(self.json)


class SavePhoto(Message):
    """Message to save photo in the database"""
    def __init__(self, identification, photo):
        self.id = identification
        self.photo = photo
        super().__init__("savePhoto")                        

class GetPhoto(Message):
    """Message to get the photo from the database"""
    def __init__(self, identification):
        self.id = identification
        super().__init__("getPhoto")

class SendPhoto(Message):
    """Message to photo to the FotoFaces"""
    def __init__(self, photo):
        self.photo = photo
        super().__init__("sendPhoto")                      


class FotoFacesProtocol:
    """FotoFaces Protocol"""

    @classmethod
    def savePhoto(cls, identification: int, photo: str) -> SavePhoto:
        """Creates a SavePhoto object."""
        save = SavePhoto(identification, photo)
        return save

    @classmethod
    def getPhoto(cls, identification: int) -> GetPhoto:
        """Creates a GetPhoto object."""
        get = GetPhoto(identification)
        return get

    @classmethod
    def sendPhoto(cls, photo: str) -> SendPhoto:
        """Creates a SendPhoto object."""
        send = SendPhoto(photo)
        return send

    @classmethod
    def send_msg(cls, connection: socket, msg: Message):
        """Sends through a connection a Message object."""
        data = json.dumps(msg.json).encode('utf-8')                 # codificar a mensagem depois de transformar em json
        header = len(data).to_bytes(2, "big")                       # header codificado em 2 bytes, big endian
        connection.sendall(header+data)                             # enviar o header + data pela socket

    @classmethod
    def recv_msg(cls, connection: socket) -> Message:
        header = int.from_bytes(connection.recv(2), "big")          # descodificar os 2 bytes, big endian para o header
        message_encoded = connection.recv(header)                   # ler o numero de bytes a que o header corresponde
        message_decoded = message_encoded.decode("utf-8")           # descodificar a data

        if len(message_decoded) == 0:                               # se nao existir data entao faz return
            return False
        try:
            message_recv = json.loads(message_decoded)              # se nao conseguir fazer load do json entao raise CDProtoBadFormat, se sim entao guarda na variavel
        except:
            raise FotoFacesProtocolBadFormat(message_encoded)

        if message_recv["command"] == "savePhoto":                       # verificar de que tipo é e criar o object desse tipo com as caracteristicas na data
            obj = FotoFacesProtocol.savePhoto(message_recv["identification"], message_recv["photo"])
        elif message_recv["command"] == "getPhoto":
            obj = FotoFacesProtocol.getPhoto(message_recv["identification"])
        elif message_recv["command"] == "sendPhoto":
            obj = FotoFacesProtocol.sendPhoto(message_recv["photo"])
        else:
            raise FotoFacesProtocolBadFormat(message_encoded)

        return obj


class FotoFacesProtocolBadFormat(Exception):
    """Exception when source message is not FotoFacesProtocol"""

    def __init__(self, original_msg: bytes = None):
        """Store original message that triggered exception."""
        self._original = original_msg

    @property
    def original_msg(self) -> str:
        """Retrieve original message as a string."""
        return self._original.decode("utf-8")