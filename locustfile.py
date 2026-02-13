from locust import HttpUser, task, between


class ApiLoadRunner(HttpUser):
    wait_time = between(0.5, 2.5)

    @task
    def request(self):
        headers = {
            "Content-Type": "application/json"
        }
        request_body = {
            "CreditScore": 600,
            "Age": 40,
            "Tenure": 3,
            "Balance": 60000,
            "NumOfProducts": 2,
            "HasCrCard": 1,
            "IsActiveMember": 1,
            "EstimatedSalary": 50000,
            "Satisfaction_Score": 3,
            "Point_Earned": 500,
            "Geography_Germany": 0,
            "Geography_Spain": 0,
            "Gender_Male": 1,
            "Gender_Female": 0,
            "Card_Type_GOLD": 1,
            "Card_Type_PLATINUM": 0,
            "Card_Type_SILVER": 0
            
        }
        self.client.post('/predict', json=request_body, headers=headers)