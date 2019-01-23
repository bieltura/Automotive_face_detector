from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import numpy as np

Base = declarative_base()
engine = create_engine('sqlite:///sqlalchemy_example.db')

class Person(Base):
    __tablename__ = 'Person'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    face_img_path = Column(String, nullable=False)
    face_features = Column(String, nullable=False)


def create_database():
    # Create an engine that stores data in the local directory's
    # sqlalchemy_example.db file.

    Base.metadata.create_all(engine)

def add_person(name, picture):

    Base.metadata.bind = engine

    DBSession = sessionmaker(bind=engine)
    session = DBSession()

    # Compute the face features of a picture
    face_features = np.array([1,2,3,4])

    new_person = Person(name=name, face_img_path=picture, face_features=face_features.tostring())
    session.add(new_person)
    session.commit()


def get_all_persons():

    Base.metadata.bind = engine

    DBSession = sessionmaker()
    DBSession.bind = engine
    session = DBSession()

    return session.query(Person).all()


def get_person_from_id(id):

    Base.metadata.bind = engine

    DBSession = sessionmaker()
    DBSession.bind = engine
    session = DBSession()

    return session.query(Person).get(id)
