def save_results_to_database(results):
    # Connect to the MySQL database
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="autoai"
    )

    cursor = connection.cursor()

    # Create table if it doesn't exist
    create_table_query = """
    CREATE TABLE IF NOT EXISTS results (
        id INT AUTO_INCREMENT PRIMARY KEY,
        model_name VARCHAR(255) NOT NULL,
        accuracy FLOAT NOT NULL,
        f1_score FLOAT NOT NULL,
        auc_score FLOAT NOT NULL
    )
    """
    cursor.execute(create_table_query)

    # Insert results into the table
    for result in results:
        insert_query = f"""
        INSERT INTO results (model_name, accuracy, f1_score, auc_score)
        VALUES (%s, %s, %s, %s)
        """
        values = (result["model_name"], result["accuracy"], result["f1_score"], result["auc_score"])
        cursor.execute(insert_query, values)

    # Commit the changes and close the connection
    connection.commit()
    cursor.close()
    connection.close()
