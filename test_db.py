from sqlalchemy import create_engine,text
import pandas as pd

# Create engine (e.g., for PostgreSQL or SQLite)
engine = create_engine('sqlite:///app.db')
# with engine.begin() as conn:
#     query=text("DELETE from interview_bookings WHERE name = Subham Parajuli.")
#     conn.execute(query)
# Execute query and load into DataFrame
df = pd.read_sql("SELECT * from interview_bookings", engine)
print(df)   