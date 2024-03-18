### Data Agent

![Example Image](images/agent.png)

Data Agent is a project designed to utilize large models to generate SQL queries for fetching data, subsequently creating corresponding reports and charts. The project architecture includes:

- `auto_plot_mat.py`: Script for generating charts, which can deal with most of the situations of the dataframe with time_column, numeric_column and category_column.
- `main.py`: Main function script.
- `data/`: Directory containing two example CSV files for testing.

### How to Use

1. **Upload Data**: Begin by uploading one or more CSV files. These files will be used for generating SQL queries and subsequent analysis.

2. **Explore Data Schema**: : Upon uploading data, the system generates a description of the schema, allowing users to understand the structure of their data for better insights and query formulation.

3. **Ask Questions**: Describe the data you want to query in the provided textbox.

4. **View Reports and Visualizations**: The system will generate SQL queries based on your description, fetch the data, debug the sql and regenerate until there is no errors, create visualizations, and present a detailed report.

### Features
- **Data Schema Description:**: Upon data upload, the system provides a detailed description of the schema, including data types and descriptions of columns. This enables users to understand their data better and formulate more precise queries.

- **Intelligent SQL Querying**: Utilizes a large language model to generate complex SQL queries based on natural language descriptions.

- **Error Correction**: In case of SQL errors, the llm can automatically debug the sql and regenerate the correct SQL until there is no errors, allowing for a smoother workflow.

- **Data Visualization**: The system automatically generates visualizations such as charts based on the fetched data.



- **Workflow**: This project is designed with a workflow in mind, ensuring a structured approach to data analysis:

  1. **Data Upload**: CSV files are uploaded for processing.

  2. **Data Schema Generation**: The system generates a schema description to help users understand the data structure.

  3. **Query Description**: Users describe the data they want to analyze in natural language.
  
  4. **SQL Query Generation**: The system generates SQL queries based on the description.
  
  5. **SQL Execution**: Queries are executed on the uploaded data to fetch relevant information.
  
  6. **Report and Visualization**: The system generates detailed reports and visualizations from the fetched data.
  
  7. **Error Handling**: If there are SQL errors, the system provides suggestions for correction.
  
  8. **Final Report**: Users receive a comprehensive report, including SQL queries used, retrieved data, and visualizations.

### Example Queries

Here are a few examples of the types of questions you can ask:

- "What is the average income of individuals with a bachelor's degree?"
- "What is the average income by geographic region?"
- "How many individuals are single?"

### How to Run

1. Ensure you have the necessary dependencies installed.
   
2. Run `main.py` with your OpenAI API key as an argument. Example:
   ```
   python main.py --openai_key YOUR_OPENAI_API_KEY
   ```

3. Follow the instructions in the Gradio interface to upload data, ask questions, and view reports.

### Dependencies

- `openai`: OpenAI Python SDK for interfacing with GPT-3.5 Turbo.
- `pandas`: For data manipulation and analysis.
- `pandasql`: For running SQL queries on DataFrame objects.
- `gradio`: For creating the user interface.

### Error Handling

- **SQL Error Correction**: If there are errors in the generated SQL queries, the system guides the user by suggesting corrected queries, ensuring a smoother workflow.


### Contact

For any inquiries or feedback, please contact Dalin Wang  at masquerlin@gmail.com.

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.