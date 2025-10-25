from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import sys
import os
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Table, MetaData
import sqlalchemy
import datetime
sys.path.append(os.path.dirname(__file__) + "./../src/Ansyas/")
import mas_autocomplete
from ansyas import Ansyas
import time
import pprint
import hashlib
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy.orm.exc import MultipleResultsFound


app = FastAPI()
temp_folder = "/opt/ansyas/temp"


origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnsyasCacheTable:
    def disconnect(self):
        self.session.close()
        
    def connect(self):
        self.engine = sqlalchemy.create_engine(f"sqlite:////{temp_folder}/cache.db", isolation_level="AUTOCOMMIT")

        Base = declarative_base()

        class AnsyasCache(Base):
            __tablename__ = 'ansyas_cache'
            hash = Column(String, primary_key=True)
            data = Column(String)
            created_at = Column(String)

        # Create all tables in the engine
        Base.metadata.create_all(self.engine)

        metadata = sqlalchemy.MetaData()
        metadata.reflect(self.engine, )
        Base = automap_base(metadata=metadata)
        Base.prepare()

        Session = sqlalchemy.orm.sessionmaker(bind=self.engine)
        self.session = Session()
        self.Table = Base.classes.plot_cache

    def insert(self, hash, data):
        try:
            self.connect()
        except sqlalchemy.exc.OperationalError:
            return False
        data = {
            'hash': hash,
            'data': data,
            'created_at': datetime.datetime.now(),
        }
        row = self.Table(**data)
        self.session.add(row)
        self.session.flush()
        self.session.commit()
        self.disconnect()
        return True

    def read(self, hash):
        try:
            self.connect()
        except sqlalchemy.exc.OperationalError:
            return None
        query = self.session.query(self.Table).filter(self.Table.hash == hash)
        try:
            data = query.one().data
        except MultipleResultsFound:
            data = None
        except NoResultFound:
            data = None
        self.disconnect()
        return data


@app.get("/")
async def root():
    return {"message": "Welcome to Ansyas API!"}


@app.post("/create_simulation_from_mas", include_in_schema=False)
async def create_magnetic_simulationion_from_mas(request: Request):
    print("Mierda 0")
    json = await request.json()

    mas = json["mas"]
    mas = mas_autocomplete.autocomplete(mas)

    hash_value = hashlib.sha256(str(mas).encode()).hexdigest()
    cache = AnsyasCacheTable()

    cached_datum = cache.read_plot(hash_value)
    if cached_datum is not None:
        print("Hit in cache!")
        return cached_datum

    print("Mierda 1")
    operating_point_index = 0
    solution_type = "EddyCurrent"
    outputs_folder = temp_folder
    project_name = f"Unnamed_design_{time.time()}"
    configuration = {
        "number_segments_arcs": 12,
        "initial_mesh_configuration": 2,
        "maximum_error_percent": 5,
        "refinement_percent": 5,
        "scale": 1,
    }

    if "operating_point_index" in json:
        operating_point_index = int(json["operating_point_index"])

    if "configuration" in json:
        configuration = json["configuration"]

    if "solution_type" in json:
        solution_type = json["solution_type"]

    if "project_name" in json:
        project_name = json["project_name"] + f"_{time.time()}"

    ansyas = Ansyas(**configuration)

    project = ansyas.create_project(
        outputs_folder=outputs_folder,
        project_name=project_name,
        # specified_version="2023.2",
        non_graphical=False,
        solution_type=solution_type,
        new_desktop_session=False
    )
    ansyas.set_units("meter")
    ansyas.create_magnetic_simulation(
        mas=mas,
        simulate=False,
        operating_point_index=operating_point_index
    )

    print(ansyas.get_project_location())
    output_project_path = ansyas.get_project_location()

    if output_project_path is None:
        raise HTTPException(status_code=418, detail="Wrong data")
    else:
        with open(output_project_path) as f:
            cache.insert_plot(hash_value, f.read())

        return FileResponse(output_project_path)
