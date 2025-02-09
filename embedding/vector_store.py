# TODO: Implement this module

module_implemented = False

class VectorBase:
    def __init__(self):
        # Not implemented, use dummy VectorBase for now, retrieving empty results
        raise NotImplementedError("VectorBase is not implemented yet.")
    def retrieve(self, query: str, top_k: int) -> list[dict]:
        # TODO: Implement this method
        # Return format:
        # [
        #     {
        #         "text": "code snippet",
        #         "metadata": {
        #             "filename": "filename",
        #             "line_number": "line number"
        #         }
        #     }
        # ] 
        
        return None
    
    def __str__(self):
        return "VectorBase"
        

class DummyVB(VectorBase):
    def __init__(self):
        # read csv file
        import pandas as pd
        data = pd.read_csv("tests/data/dummyVB/Fake_Python_Math_Functions.csv")
        self.data = [{"text": row["text"], "metadata": eval(row["metadata"])} for _, row in data.iterrows()]
        print("Dummy VectorBase initialized.")
    def retrieve(self, query: str, top_k: int) -> list[dict]:
        return self.data
    
    def __str__(self):
        return "Dummy VectorBase"