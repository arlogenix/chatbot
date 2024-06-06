import os
import requests
import pytz
from datetime import datetime
import traceback
import logging

from flask import Flask, request, jsonify
from flask_cors import CORS

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()


app = Flask(__name__)
CORS(app)

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')


def get_matched_stations(contact_number=None, start_station_name=None, end_station_name=None, service=None):
    url = "https://api.onlinetransport.co.za/v1/json/private/match-stations/"
    params = {
        "pncontact": contact_number,
        "pstart": start_station_name,
        "pend": end_station_name,
        "pservice": service,
        "apiKey": "eb38d7bc-e510-11ee-80e7-3d055a162a32"
    }

    response = requests.get(url, params=params)

    return response.json()


def get_day_number():
    south_africa_tz = pytz.timezone('Africa/Johannesburg')
    south_africa_time = datetime.now(south_africa_tz)
    weekday = south_africa_time.weekday()
    day_number = (weekday + 1) % 7 + 1
    return day_number


def get_timing(input_time=None):
    south_africa_tz = pytz.timezone('Africa/Johannesburg')
    current_datetime = datetime.now(south_africa_tz)

    if input_time is None:
        input_datetime = current_datetime
    else:
        try:
            input_datetime = datetime.fromisoformat(input_time)
        except ValueError:
            current_date = current_datetime.date()
            try:
                input_time_obj = datetime.strptime(input_time, "%H:%M").time()
            except ValueError:
                input_time_obj = datetime.strptime(input_time, "%I:%M %p").time()
            input_datetime = datetime.combine(current_date, input_time_obj)
        
        if input_datetime.tzinfo is None:
            input_datetime = south_africa_tz.localize(input_datetime)
        else:
            input_datetime = input_datetime.astimezone(south_africa_tz)

    formatted_time = input_datetime.strftime('%d/%m/%Y %H:%M')
    return formatted_time

@tool
def get_transport_schedule(pickup_station_name=None, destination_station_name=None, service=None, time=None):
    """Look up information about transport schedule of bus, train and flight"""

    if not pickup_station_name:
        return {"error": "Please provide pickup station name"}
    if not destination_station_name:
        return {"error": "Please provide destination station name"}
    if not service:
        return {"error": "Please provide transport mode"}
    
    
    match_stations = get_matched_stations('', pickup_station_name, destination_station_name, service)

    plat = match_stations[0]['start_latitude']
    plon = match_stations[0]['start_longitude']
    dlat = match_stations[0]['end_latitude']
    dlon = match_stations[0]['end_longitude']
    days = get_day_number()
    timing = get_timing(time)

    if service.lower() == 'train':
        url = "https://api.onlinetransport.co.za/v1/online-train-service/directions/train-schedule/"
    elif service.lower() == 'bus':
        url = "https://api.onlinetransport.co.za/v1/online-transit-service/directions/bus-schedule/"
    elif service.lower() == 'flight':
        url = "https://api.onlinetransport.co.za/v1/online-flight-service/directions/flight-schedule/"

    params = {
        "plat": plat,
        "plon": plon,
        "dlat": dlat,
        "dlon": dlon,
        "timing": timing,
        "days": days,
        "options": 1,
        "provider": "",
        "apiKey": "eb38d7bc-e510-11ee-80e7-3d055a162a32"
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Request failed with status code {response.status_code}"}

def format_for_whatsapp(text):
    formatted_text = text.replace("**", "*")
    return formatted_text

chat_history = []
@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()

        if 'question' not in data:
            return jsonify({'error': 'Missing question parameter'}), 400

        question = data['question']

        llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

        tools = [get_transport_schedule]

        MEMORY_KEY = "chat_history"
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an transport assistant for question answering tasks. Use the following excerpts of retrieved context to answer the question.
                    If you don't know the answer just say that 'Could you please let us know if you require train, bus or flight services? We are here for you!'.
                    """,
                ),
                MessagesPlaceholder(variable_name=MEMORY_KEY),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        llm_with_tools = llm.bind_tools(tools)

        agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                    x["intermediate_steps"]
                ),
                "chat_history": lambda x: x["chat_history"],
            }
            | prompt
            | llm_with_tools
            | OpenAIToolsAgentOutputParser()
        )
        agent_executor = AgentExecutor(agent=agent, tools=tools, return_intermediate_steps=True, verbose=True)

        result = agent_executor.invoke({"input": question, "chat_history": chat_history})
        
        if len(chat_history) > 6:
            chat_history.clear()
            
        chat_history.extend(
            [
                HumanMessage(content=question),
                AIMessage(content=result["output"]),
            ]
        )

        answer = result["output"]
        
        api_data = None
        if result.get('intermediate_steps', [{}]):
            api_data = result.get('intermediate_steps', [{}])[0]
            api_data = api_data[1]

        
        
        formatted_answer = format_for_whatsapp(answer)

        response = {
            "question": question,
            "answer": formatted_answer,
            "data": api_data
        }

        return jsonify(response), 200
    except Exception as e:
        error_message = str(e)
        traceback_str = traceback.format_exc()
        logging.error(f"Error occurred: {error_message}\n{traceback_str}")
        return jsonify({'error': 'An error occurred', 'message': error_message, 'traceback': traceback_str}), 500

if __name__ == '__main__':
    app.run(debug=True)
