### **LLM Based orchestration workflow for finding the optimal path between two locatons given in natural language**

#### **current model:mistal-7B**

this library extracts the source and destinatipn information from the LLM and then finds the optimal path between the 
two points using reinforcement learning 

#### **Setup instructons**

1. install requirements

    ```pip install -U -r requirements.txt```

2. install mistral locally using ollama

    2.1 instal ollama

    ```curl https://ollama.ai/install.sh | sh ```

    2.2 add mistal-7b as LLM

    ```ollama serve & ollama pull mistral```

    2.3 check if the model is added
    
    ```ollama list ```

3. install uvicorn

    ```sudo apt install uvicorn```

4. run the api code

    ```uvicorn main:app --reload```

5. call api to get cordinates

    ``` 
         curl -X POST "http://127.0.0.1:8000/prompt"      
         -H "Content-Type: application/json"      
         -d '{"text": "hi cameraman go to lot-b1-zone-16-rack-a1 and pickup a box marked as skibdi toilet and bring it to the docking bay"}'
    ```

6. one shot prompt 

imagine that you are a robot navigating a warehouse in order to do so you need to find the optimal path between the starting and ending locations" \
you will follow the below steps to find the optmal path.

Here's the optimal path:

Step 1: Get the cordinates for the given locations
Action: GetCordinates

Step 2: Calculate the optimal path
Action: Get Optimal Paths

7. call api to get optimal path

```
    curl -X 'POST' \
    'http://127.0.0.1:8000/prompt/optimal_path' \
    -H 'Content-Type: application/json' \
    -d '{
        "start_position": {"x": "12", "y": "34"},
        "stop_position": {"x": "56", "y": "78"}
    }'
```

8. to run the agent code 

``` python3 run.py ```

enter the prompt and the optimaal path will be returned

