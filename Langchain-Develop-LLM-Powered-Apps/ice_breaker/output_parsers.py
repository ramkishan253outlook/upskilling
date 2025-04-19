from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Dict, Any

class Summary(BaseModel):
    summary: str = Field(description="summary")
    facts: list[str] = Field(description="Instresting facts about the person")

    def to_dict(self) -> Dict[str, Any]:
        return {"summary": self.summary, "facts": self.facts}


summary_parser = PydanticOutputParser(pydantic_object=Summary)   
