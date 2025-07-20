import csv
import random

def test_data_loading():
    """Test function to verify the used car price dataset can be loaded correctly."""
    
    all_x = []
    all_y = []
    
    try:
        with open('dataset/cleaned_student_performance_data.csv', 'r') as file:
            reader = csv.reader(file)
            header = next(reader)
            data = list(reader)
            print(f"Data size: {len(data)}")
            print(f"Header: {header}")
            
            random.shuffle(data)
            for r in data:
                all_x.append([float(i) for i in r[:-1]])  # All features except the last one (price)
                all_y.append(float(r[-1]))  # Last column is the target (price)
        
        print(f"Input size: {len(all_x[0])}")
        print(f"Sample input: {all_x[0]}")
        print(f"Sample target: {all_y[0]}")
        
        # Check data ranges
        min_price = min(all_y)
        max_price = max(all_y)
        print(f"Price range: ${min_price:.2f} - ${max_price:.2f}")
        
        # Check feature ranges for the first few continuous features
        make_years = [x[0] for x in all_x]
        mileage = [x[1] for x in all_x]
        engine_cc = [x[2] for x in all_x]
        
        print(f"Make year range: {min(make_years)} - {max(make_years)}")
        print(f"Mileage (kmpl) range: {min(mileage):.2f} - {max(mileage):.2f}")
        print(f"Engine CC range: {min(engine_cc)} - {max(engine_cc)}")
        
        print("Data loading successful!")
        return True
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return False

if __name__ == "__main__":
    test_data_loading() 