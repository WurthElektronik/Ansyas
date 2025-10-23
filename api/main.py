from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse
import base64
import sys
import os
sys.path.append(os.path.dirname(__file__) + "./../src/Ansyas/")
import mas_autocomplete
from ansyas import Ansyas
import time
import pprint

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


@app.get("/")
async def root():
    return {"message": "Welcome to Ansyas API!"}


@app.post("/create_simulation_from_mas", include_in_schema=False)
async def create_simulation_from_mas(request: Request):
    json = await request.json()

    mas = json["mas"]
    mas = mas_autocomplete.autocomplete(mas)
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
        raise HTTPException(status_code=418, detail="Wrong dimensions")
    else:
        return FileResponse(output_project_path)
