# src/data_loader.py

import psycopg2
import numpy as np

def load_data_from_postgresql():
    conn = psycopg2.connect(dbname="dreamseeker",
                            user="glizkmoe.fit",
                            password="<PASSWORD>",
                            host="localhost",
                            port="5432"
    )
    cur = conn.cursor()

    # Query all rows
    cur.execute("SELECT * FROM bucket_list_activities;")
    rows = cur.fetchall()

    # Close connection
    cur.close()
    conn.close()

    # Optional - Converted to NumPy array
    data = np.array(rows)

    # Split into features(X) and label/outcome variable/dependent variable (y)
    # label is the last column
    X = data[:, :-1]#all columns except the last column
    y = data[:, -1] # Last column (label)

    X = np.array(X) #convert to Numpy array
    y = np.array(y) .astype(int) # convert labels to numpy array, integers

    return X, y

if __name__ == "__main__":
    X, y = load_data_from_postgresql()

    print("--- Loaded Features (X) ---")
    print(X)
    print("\nShape of X:", X.shape)

    print("\n--- Loaded Labels (y) ---")
    print(y)
    print("\nShape of y:", y.shape)
