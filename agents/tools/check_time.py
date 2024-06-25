from datetime import datetime
from langchain.tools import Tool


def check_time():
    return datetime.now()


run_query_tool = Tool.from_function(
    name="check_time",
    description="checks the current datetime in UTC format",
    func=check_time,
)
