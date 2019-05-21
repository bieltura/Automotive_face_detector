from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

# Database setup and metadata
Base = declarative_base()
engine = create_engine('sqlite:///database/IDNEO.db')
Base.metadata.bind = engine

DBSession = sessionmaker()
DBSession.bind = engine
session = DBSession()


class Person(Base):
    __tablename__ = 'Person'

    # Person features
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    face_img_path = Column(String, nullable=False)
    face_features = Column(String, nullable=False)


def create_database():
    Base.metadata.create_all(engine)


def add_person(name, picture_path, face_features):

    # Compute the face features of a picture
    new_person = Person(name=name, face_img_path=picture_path, face_features=face_features.tostring())
    session.add(new_person)
    session.commit()


def get_all_persons():
    return session.query(Person).all()


def get_person_from_id(id):
    return session.query(Person).get(id)


def get_persons_by_name(name):
    # Asume there is only one person with the same name
    return session.query(Person).filter_by(name=name).first()
