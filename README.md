# SQL Agent Chatbot

A powerful conversational AI chatbot designed to transform natural language questions into optimized SQL queries, offering **real-time employee insights** from a MySQL database along with **dynamic visual analytics** like charts and graphs.

## 🔍 Features

- ✅ Interprets **natural language** queries.
- ✅ Generates **dynamic and optimized MySQL queries**.
- ✅ Fetches real-time data from the **employee database**.
- ✅ Responds with **natural, human-like answers**.
- ✅ Automatically generates **visualizations** like bar charts and pie charts for analytics.
- ✅ Uses **LangChain, OpenAI GPT, and MySQL** integration.

## 🧠 Technologies Used

- **Python** 🐍
- **LangChain**
- **OpenAI GPT-4o**
- **MySQL**
- **SQLAlchemy**
- **Matplotlib**
- **Pandas**

## ⚙️ System Architecture

1. **LangChain SQL Agent** with OpenAI LLM (GPT-4o).
2. **MySQL Database** (`database_name`) with employee-related tables.
3. **SQLDatabaseToolkit** used for query generation.
4. **Custom Prompt Template** guides the AI assistant to:
   - Answer queries.
   - Suggest chart types and generate visual insights.

## 📊 Capabilities

| Functionality | Description |
|---------------|-------------|
| Employee Info | `What is the total number of employees in each department?` |
| Utilization   | `Show me the utilization of Ram from last 4 weeks.` |
| Charts        | Automatically plots graphs (e.g., pie chart for department-wise distribution) |
| Chat Memory   | Retains recent conversation context using `ConversationBufferMemory` |

## 🚀 How to Run

1. **Clone the repository** and open the Jupyter Notebook.
2. **Set up MySQL DB** and update credentials:
   ```python
   mysql_user = "root"
   mysql_password = "your_password"
   mysql_db_name = "database_name"
   ```
3. **Insert your OpenAI API key - Paid**:
   ```python
   llm = ChatOpenAI(model="gpt-4o", openai_api_key="your_key")
   ```
   **or**

   **Insert your GROQ API key - Free**:
   ```python
   os.environ["GROQ_API_KEY"] = "your_key"
   llm = ChatGroq(
       model="llama3-70b-8192",
       temperature=0,
       max_tokens=1024,
   )
   ```
   
5. **Run cells sequentially** and interact with the chatbot using plain English queries.

## 📷 Sample Use Cases

- 🧑‍💼 “How many employees joined in 2024?”
- 🏢 “Generate a bar chart for employee count by department.”
- 🕒 “What is the average number of hours employees worked last week?”

## 🧪 Future Enhancements

- Integration with **Streamlit** or **FastAPI** for UI.
- Enhanced **security and authentication**.
- Exportable **PDF reports**.
- Auto-email of scheduled reports.

## 🤝 Contributors

- **Aman Chaturvedi**  
  [LinkedIn](https://www.linkedin.com/in/aman8333) | [GitHub](https://github.com/divyam8333)
