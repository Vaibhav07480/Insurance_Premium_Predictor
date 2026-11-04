from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, computed_field
from typing import Literal, Annotated

from predictor import (
    age_group as get_age_group,
    bmi as calc_bmi,
    city_tier as get_city_tier,
    lifestyle_risk as get_lifestyle_risk,
    predict_premium_category,
)

app = FastAPI()


class UserInput(BaseModel):
    age: Annotated[int, Field(..., gt=0, lt=120, description="Age of the user")]
    weight: Annotated[float, Field(..., gt=0, description="Weight of the user")]
    height: Annotated[float, Field(..., gt=0, lt=2.5, description="Height of the user")]
    income_lpa: Annotated[float, Field(..., gt=0, description="Annual salary of the user in lpa")]
    smoker: Annotated[bool, Field(..., description="Is user a smoker")]
    city: Annotated[str, Field(..., description="The city that the user belongs to")]
    occupation: Annotated[
        Literal[
            "retired",
            "freelancer",
            "student",
            "government_job",
            "business_owner",
            "unemployed",
            "private_job",
        ],
        Field(..., description="Occupation of the user"),
    ]

    @computed_field
    @property
    def bmi(self) -> float:
        return calc_bmi(self.weight, self.height)

    @computed_field
    @property
    def lifestyle_risk(self) -> str:
        return get_lifestyle_risk(self.smoker, self.bmi)

    @computed_field
    @property
    def age_group(self) -> str:
        return get_age_group(self.age)

    @computed_field
    @property
    def city_tier(self) -> int:
        return get_city_tier(self.city)


@app.post("/predict")
def predict_premium(data: UserInput):
    prediction = predict_premium_category(
        age=data.age,
        weight=data.weight,
        height=data.height,
        income_lpa=data.income_lpa,
        smoker=data.smoker,
        city=data.city,
        occupation=data.occupation,
    )
    return JSONResponse(status_code=200, content={"predicted_category": prediction})
