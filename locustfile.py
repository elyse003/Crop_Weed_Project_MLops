from locust import HttpUser, task, between

class MLUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def predict_endpoint(self):
        # Ensure you have a dummy image named 'test.jpg' in the same folder
        # or catch the error if file is missing
        try:
            with open("ridderzuring_3109_jpg.rf.b8e99770acf95370b238f2af2a71f5a3.jpg", "rb") as image:
                self.client.post("/predict", files={"file": image})
        except FileNotFoundError:
            print("test.jpg not found for locust testing")