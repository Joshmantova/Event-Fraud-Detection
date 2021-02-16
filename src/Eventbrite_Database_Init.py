import psycopg2

host = "eventbritepostgresdb.chr629x8cxxi.us-west-2.rds.amazonaws.com"
user = "postgres"
password = "password"

conn = psycopg2.connect(host=host, 
                        user=user, 
                        password=password)
conn.autocommit = True #needed to create database

# create_database_query = "CREATE  database EventbriteData"
database_name = "eventbritedata"
create_database_query = f"CREATE database {database_name}"
if_exists_query = f"SELECT datname FROM pg_database WHERE datname='{database_name}'"
cur = conn.cursor()
cur.execute(if_exists_query)
query_out = cur.fetchall()
if query_out==[]:
    cur.execute(create_database_query)
    print('Database successfully created!')
else:
    print('Database already exists')

table_name = "eventbriteclean"
create_cleaned_table_query = f"""CREATE TABLE IF NOT EXISTS {table_name} (
    event_id INT NOT NULL
    ,body_length INT
    ,country VARCHAR(3)
    ,currency VARCHAR(3)
    ,delivery_method FLOAT
    ,fb_published INT
    ,has_analytics INT
    ,has_logo INT
    ,listed VARCHAR(1)
    ,name_length INT
    ,num_payouts INT
    ,payout_type VARCHAR(50)
    ,show_map INT
    ,user_age INT
    ,venue_country VARCHAR(2)
    ,venue_state VARCHAR(50)
    ,avg_ticket_price FLOAT
    ,total_tickets_available INT
    ,number_of_payouts INT
    ,avg_quantity_sold INT
    ,payee_name_in_org_name VARCHAR(50)
    ,venue_country_is_source_country VARCHAR(50)
    ,days_to_event INT
    ,event_length INT
    ,pop_country VARCHAR(50)
)"""
cur.execute(create_cleaned_table_query)
#create true label and prediction table
table_name = "labelsandpredictions"
create_label_prediction_table_query = f"""
CREATE TABLE IF NOT EXISTS {table_name} (
    EventID INT NOT NULL
    ,true_label INT
    ,model_prediction INT
)
"""
cur.execute(create_label_prediction_table_query)

#create raw data table
table_name = "rawdata"
create_raw_data_table_query = f"""
CREATE TABLE IF NOT EXISTS {table_name} (
    event_id INT NOT NULL
    ,approx_payout_date BIGINT
    ,body_length INT
    ,channels INT
    ,country VARCHAR
    ,currency VARCHAR
    ,delivery_method INT
    ,description VARCHAR
    ,email_domain VARCHAR
    ,event_created BIGINT
    ,event_end BIGINT
    ,event_published BIGINT
    ,event_start BIGINT
    ,fb_published INT
    ,gts FLOAT
    ,has_analytics INT
    ,has_header VARCHAR
    ,has_logo INT
    ,listed VARCHAR
    ,name VARCHAR
    ,name_length INT
    ,num_order INT
    ,num_payouts INT
    ,object_id BIGINT
    ,org_desc VARCHAR
    ,org_facebook INT
    ,org_name VARCHAR
    ,org_twitter INT
    ,payee_name VARCHAR
    ,payout_type VARCHAR
    ,previous_payouts VARCHAR
    ,sale_duration INT
    ,sale_duration2 INT
    ,show_map INT
    ,ticket_types VARCHAR
    ,user_age INT
    ,user_create BIGINT
    ,user_type INT
    ,venue_address VARCHAR
    ,venue_country VARCHAR
    ,venue_latitude FLOAT
    ,venue_longitude FLOAT
    ,venue_name VARCHAR
    ,venue_state VARCHAR
)
"""
cur.execute(create_raw_data_table_query)