import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, r2_score, silhouette_score, mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
import re
import io
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try importing LLM libraries
try:
    import openai
    from openai import OpenAI
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Chat with Your Data 🤖",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        background: #f8f9fa;
        margin: 1rem 0;
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.8rem;
        border-radius: 15px 15px 5px 15px;
        margin: 0.5rem 0;
        margin-left: 20%;
        word-wrap: break-word;
    }
    .ai-message {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #333;
        padding: 0.8rem;
        border-radius: 15px 15px 15px 5px;
        margin: 0.5rem 0;
        margin-right: 20%;
        word-wrap: break-word;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .status-success {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 0.5rem;
        border-radius: 5px;
        color: white;
        margin: 0.5rem 0;
    }
    .status-error {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        padding: 0.5rem;
        border-radius: 5px;
        color: white;
        margin: 0.5rem 0;
    }
    .feature-card {
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        background: #f8f9fa;
        margin: 0.5rem 0;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class DataChatbot:
    def __init__(self, df, api_key=None):
        self.df = df
        self.api_key = api_key
        self.client = None
        self.is_ready = False
        self.conversation_history = []
        self.initialization_attempted = False
        
        # Only initialize if we have a valid API key
        if api_key and api_key.strip() and LLM_AVAILABLE:
            self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client with better error handling"""
        if self.initialization_attempted:
            return self.is_ready
            
        self.initialization_attempted = True
        
        try:
            # Validate API key format first
            api_key = self.api_key.strip()
            if not api_key.startswith('sk-'):
                return False
                
            # Initialize client
            self.client = OpenAI(api_key=api_key)
            
            # Test with a very simple request
            test_response = self.client.models.list()
            
            self.is_ready = True
            return True
            
        except Exception as e:
            self.is_ready = False
            self.client = None
            # Don't show error here, handle it in the UI
            return False
    
    def get_data_context(self):
        """Create a comprehensive data context for the AI"""
        if self.df is None:
            return "No data available."
        
        # Basic info
        basic_info = f"""
Dataset Overview:
- Shape: {self.df.shape[0]} rows, {self.df.shape[1]} columns
- Memory usage: {self.df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB
"""
        
        # Column information
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        column_info = f"""
Column Information:
- Numeric columns ({len(numeric_cols)}): {', '.join(numeric_cols[:10])}{'...' if len(numeric_cols) > 10 else ''}
- Categorical columns ({len(categorical_cols)}): {', '.join(categorical_cols[:10])}{'...' if len(categorical_cols) > 10 else ''}
"""
        
        # Missing values
        missing_info = ""
        missing_data = self.df.isnull().sum()
        if missing_data.sum() > 0:
            missing_cols = missing_data[missing_data > 0]
            missing_info = f"\nMissing Values:\n{missing_cols.to_string()}"
        
        # Statistical summary for numeric columns
        stats_info = ""
        if len(numeric_cols) > 0:
            stats_df = self.df[numeric_cols].describe()
            stats_info = f"\nNumerical Statistics (first 5 columns):\n{stats_df.iloc[:, :5].to_string()}"
        
        # Sample data
        sample_info = f"\nSample Data (first 3 rows):\n{self.df.head(3).to_string()}"
        
        return basic_info + column_info + missing_info + stats_info + sample_info
    
    def chat(self, user_message):
        """Enhanced chat function with better error handling"""
        # If not ready, try to initialize first
        if not self.is_ready and self.api_key:
            self._initialize_client()
            
        if not self.is_ready:
            return self._fallback_response(user_message)
        
        try:
            # Add user message to history
            self.conversation_history.append({"role": "user", "content": user_message})
            
            # Create system message with data context
            system_message = f"""
You are an expert data analyst chatbot. You help users understand and analyze their dataset through conversation.

Current Dataset Context:
{self.get_data_context()}

Guidelines:
1. Provide specific, actionable insights about the data
2. Suggest visualizations when relevant
3. Explain statistical concepts in simple terms
4. Always reference specific columns and values from the dataset
5. If asked about creating charts, describe what type would be best and why
6. Keep responses conversational but informative
7. If you need to perform calculations, show your reasoning

Previous conversation context is maintained for follow-up questions.
"""
            
            # Prepare messages for API call
            messages = [{"role": "system", "content": system_message}]
            
            # Add recent conversation history (last 6 messages to avoid token limits)
            recent_history = self.conversation_history[-6:]
            messages.extend(recent_history)
            
            # Make API call with proper error handling
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=800,
                temperature=0.7
            )
            
            ai_response = response.choices[0].message.content
            
            # Add AI response to history
            self.conversation_history.append({"role": "assistant", "content": ai_response})
            
            return ai_response
            
        except Exception as e:
            error_msg = str(e).lower()
            if "authentication" in error_msg or "api key" in error_msg:
                self.is_ready = False
                return "❌ Invalid API key. Please check your OpenAI API key and try again."
            elif "rate limit" in error_msg:
                return "❌ Rate limit exceeded. Please wait a moment before asking another question."
            elif "quota" in error_msg:
                return "❌ API quota exceeded. Please check your OpenAI account billing."
            else:
                return self._fallback_response(user_message)
    
    def _fallback_response(self, user_message):
        """Provide basic analysis without AI"""
        message_lower = user_message.lower()
        
        if any(word in message_lower for word in ['summary', 'overview', 'describe']):
            return self._generate_summary()
        elif any(word in message_lower for word in ['correlation', 'correlate', 'relationship']):
            return self._analyze_correlations()
        elif any(word in message_lower for word in ['missing', 'null', 'empty']):
            return self._analyze_missing_data()
        elif any(word in message_lower for word in ['column', 'columns', 'variable']):
            return self._describe_columns()
        else:
            return f"""
I'd love to provide detailed insights about your data! To enable advanced AI analysis, please add your OpenAI API key in the sidebar.

For now, here's what I can tell you:
- Your dataset has {self.df.shape[0]:,} rows and {self.df.shape[1]} columns
- You can ask me about: data summary, correlations, missing values, or column descriptions

Try asking: "Give me a data summary" or "Show me correlations"
"""
    
    def _generate_summary(self):
        """Generate basic data summary"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        summary = f"""
📊 **Dataset Summary:**

**Basic Info:**
- {self.df.shape[0]:,} rows and {self.df.shape[1]} columns
- {len(numeric_cols)} numeric columns
- {len(categorical_cols)} categorical columns

**Data Quality:**
- {self.df.duplicated().sum()} duplicate rows
- {self.df.isnull().sum().sum()} missing values total

**Memory Usage:** {self.df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB
"""
        
        if len(numeric_cols) > 0:
            summary += f"\n**Numeric Columns:** {', '.join(numeric_cols[:5])}{'...' if len(numeric_cols) > 5 else ''}"
        
        if len(categorical_cols) > 0:
            summary += f"\n**Categorical Columns:** {', '.join(categorical_cols[:5])}{'...' if len(categorical_cols) > 5 else ''}"
        
        return summary
    
    def _analyze_correlations(self):
        """Analyze correlations between numeric columns"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return "❌ Need at least 2 numeric columns for correlation analysis."
        
        corr_matrix = self.df[numeric_cols].corr()
        
        # Find strongest correlations
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                corr_val = corr_matrix.iloc[i, j]
                if not np.isnan(corr_val):
                    corr_pairs.append((abs(corr_val), corr_val, col1, col2))
        
        corr_pairs.sort(reverse=True)
        
        result = "📈 **Correlation Analysis:**\n\n"
        
        if corr_pairs:
            result += "**Strongest Correlations:**\n"
            for abs_corr, corr, col1, col2 in corr_pairs[:5]:
                strength = "Strong" if abs_corr > 0.7 else "Moderate" if abs_corr > 0.3 else "Weak"
                direction = "positive" if corr > 0 else "negative"
                result += f"- {col1} ↔ {col2}: {corr:.3f} ({strength} {direction})\n"
        else:
            result += "No significant correlations found."
        
        return result
    
    def _analyze_missing_data(self):
        """Analyze missing data patterns"""
        missing_data = self.df.isnull().sum()
        missing_cols = missing_data[missing_data > 0]
        
        if len(missing_cols) == 0:
            return "✅ **Great news!** Your dataset has no missing values."
        
        total_rows = len(self.df)
        result = f"🔍 **Missing Data Analysis:**\n\n"
        result += f"**Total missing values:** {missing_data.sum():,}\n\n"
        result += "**Missing by column:**\n"
        
        for col, count in missing_cols.head(10).items():
            percentage = (count / total_rows) * 100
            result += f"- {col}: {count:,} ({percentage:.1f}%)\n"
        
        if len(missing_cols) > 10:
            result += f"... and {len(missing_cols) - 10} more columns with missing data"
        
        return result
    
    def _describe_columns(self):
        """Describe dataset columns"""
        result = f"📋 **Column Information:**\n\n"
        result += f"**Total columns:** {len(self.df.columns)}\n\n"
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        if len(numeric_cols) > 0:
            result += f"**Numeric Columns ({len(numeric_cols)}):**\n"
            for col in numeric_cols[:10]:
                result += f"- {col}\n"
            if len(numeric_cols) > 10:
                result += f"... and {len(numeric_cols) - 10} more\n"
        
        if len(categorical_cols) > 0:
            result += f"\n**Categorical Columns ({len(categorical_cols)}):**\n"
            for col in categorical_cols[:10]:
                unique_count = self.df[col].nunique()
                result += f"- {col} ({unique_count} unique values)\n"
            if len(categorical_cols) > 10:
                result += f"... and {len(categorical_cols) - 10} more\n"
        
        return result

class AdvancedDataProcessor:
    def __init__(self):
        self.df = None
        self.original_df = None
        self.numeric_cols = []
        self.categorical_cols = []
        self.datetime_cols = []
        self.cleaning_log = []
        self.data_profile = {}

    def load_data(self, file):
        """Load data from uploaded file with enhanced error handling"""
        try:
            if file.name.endswith('.csv'):
                encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                for encoding in encodings:
                    try:
                        file.seek(0)
                        self.df = pd.read_csv(file, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise ValueError("Could not decode CSV file")
            elif file.name.endswith(('.xlsx', '.xls')):
                self.df = pd.read_excel(file)
            else:
                st.error("Unsupported file format")
                return False

            self.original_df = self.df.copy()
            self._analyze_columns()
            self._generate_data_profile()
            return True
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False

    def _analyze_columns(self):
        """Enhanced column analysis"""
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        # Detect datetime columns
        for col in self.categorical_cols.copy():
            sample_data = self.df[col].dropna().head(100)
            try:
                pd.to_datetime(sample_data, infer_datetime_format=True)
                self.datetime_cols.append(col)
                self.categorical_cols.remove(col)
            except:
                # Check for potential numeric columns stored as strings
                try:
                    pd.to_numeric(sample_data.str.replace(',', '').str.replace('$', ''))
                    self.df[col] = pd.to_numeric(self.df[col].str.replace(',', '').str.replace('$', ''), errors='coerce')
                    self.numeric_cols.append(col)
                    self.categorical_cols.remove(col)
                except:
                    pass

    def _generate_data_profile(self):
        """Generate comprehensive data profile"""
        self.data_profile = {
            'shape': self.df.shape,
            'memory_usage': self.df.memory_usage(deep=True).sum(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'duplicates': self.df.duplicated().sum(),
            'numeric_summary': self.df[self.numeric_cols].describe().to_dict() if self.numeric_cols else {},
            'categorical_summary': {col: self.df[col].value_counts().head(5).to_dict() 
                                  for col in self.categorical_cols[:5]}
        }

def create_visualization_from_description(df, description):
    """Create visualizations based on AI suggestions"""
    description_lower = description.lower()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    if 'correlation' in description_lower and len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr()
        fig = px.imshow(corr_matrix, title="Correlation Heatmap", color_continuous_scale="RdBu_r")
        return fig
    elif 'distribution' in description_lower and len(numeric_cols) > 0:
        col = numeric_cols[0]
        fig = px.histogram(df, x=col, title=f"Distribution of {col}", marginal="box")
        return fig
    elif 'scatter' in description_lower and len(numeric_cols) >= 2:
        fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], 
                        title=f"Scatter: {numeric_cols[0]} vs {numeric_cols[1]}")
        return fig
    
    return None

def main():
    st.markdown('<h1 class="main-header">💬 Chat with Your Data</h1>', unsafe_allow_html=True)
    st.markdown("### 🤖 AI-Powered Conversational Data Analysis")

    # Initialize session state
    if 'data_processor' not in st.session_state:
        st.session_state.data_processor = AdvancedDataProcessor()
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""

    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API Key section
        st.subheader("🤖 AI Settings")
        if LLM_AVAILABLE:
            api_key_input = st.text_input(
                "OpenAI API Key", 
                value="",
                type="password",
                placeholder="sk-...",
                help="Required for AI chat features. Get your key from https://platform.openai.com/api-keys"
            )
            
            # Only update if there's a real change and we have data
            if api_key_input and api_key_input != st.session_state.api_key:
                st.session_state.api_key = api_key_input
                if st.session_state.data_processor.df is not None:
                    # Create new chatbot instance
                    st.session_state.chatbot = DataChatbot(
                        st.session_state.data_processor.df, api_key_input
                    )
            
            # Show API key status with better validation
            if api_key_input:
                if not api_key_input.startswith('sk-'):
                    st.markdown('<div class="status-error">❌ Invalid API Key Format (should start with sk-)</div>', unsafe_allow_html=True)
                elif len(api_key_input) < 20:
                    st.markdown('<div class="status-error">❌ API Key too short</div>', unsafe_allow_html=True)
                elif st.session_state.chatbot and st.session_state.chatbot.is_ready:
                    st.markdown('<div class="status-success">🤖 AI Chat Ready!</div>', unsafe_allow_html=True)
                elif st.session_state.chatbot and st.session_state.chatbot.initialization_attempted:
                    st.markdown('<div class="status-error">❌ API Key Authentication Failed</div>', unsafe_allow_html=True)
                    with st.expander("💡 Troubleshooting"):
                        st.markdown("""
                        **Common issues:**
                        - API key is incorrect or expired
                        - No OpenAI credits available
                        - Network connection issues
                        - Rate limits exceeded
                        
                        **Solutions:**
                        - Double-check your API key from OpenAI dashboard
                        - Ensure you have billing set up
                        - Try generating a new API key
                        - Wait a few minutes if rate limited
                        """)
                else:
                    st.markdown('<div class="status-error">⏳ Validating API Key...</div>', unsafe_allow_html=True)
            else:
                st.info("💡 Add your OpenAI API key to enable AI chat")
        else:
            st.error("❌ OpenAI package not installed")
            st.code("pip install openai")

        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader("Upload your dataset", type=['csv', 'xlsx', 'xls'])

        if uploaded_file and st.button("🚀 Load Data"):
            with st.spinner("Loading data..."):
                if st.session_state.data_processor.load_data(uploaded_file):
                    st.success("✅ Data loaded successfully!")
                    # Initialize chatbot with loaded data and current API key
                    current_api_key = st.session_state.get('api_key', '')
                    if current_api_key:
                        st.session_state.chatbot = DataChatbot(
                            st.session_state.data_processor.df, 
                            current_api_key
                        )
                    else:
                        st.session_state.chatbot = DataChatbot(
                            st.session_state.data_processor.df
                        )
                    # Clear previous chat history
                    st.session_state.chat_history = []
                    st.rerun()

        # Quick actions
        if st.session_state.data_processor.df is not None:
            st.header("⚡ Quick Actions")
            if st.button("📊 Data Summary"):
                if st.session_state.chatbot:
                    response = st.session_state.chatbot.chat("Give me a comprehensive summary of this dataset")
                    st.session_state.chat_history.append(("📊 Data Summary", response))
                    st.rerun()
            
            if st.button("🔍 Find Insights"):
                if st.session_state.chatbot:
                    response = st.session_state.chatbot.chat("What are the most interesting insights and patterns in this data?")
                    st.session_state.chat_history.append(("🔍 Find Insights", response))
                    st.rerun()

    # Main content
    if st.session_state.data_processor.df is not None:
        df = st.session_state.data_processor.df
        
        # Display dataset metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        metrics = [
            (len(df), "Rows"),
            (len(df.columns), "Columns"), 
            (len(df.select_dtypes(include=[np.number]).columns), "Numeric"),
            (len(df.select_dtypes(include=['object']).columns), "Categorical"),
            (df.isnull().sum().sum(), "Missing")
        ]
        
        for col, (value, label) in zip([col1, col2, col3, col4, col5], metrics):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{value:,}</h3>
                    <p>{label}</p>
                </div>
                """, unsafe_allow_html=True)

        # Chat Interface
        st.header("💬 Chat with Your Data")
        
        # Display chat history
        if st.session_state.chat_history:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            for user_msg, ai_msg in st.session_state.chat_history:
                st.markdown(f'<div class="user-message">👤 {user_msg}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="ai-message">🤖 {ai_msg}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Chat input
        col1, col2 = st.columns([4, 1])
        with col1:
            user_input = st.text_input(
                "Ask about your data:",
                placeholder="e.g., What patterns do you see? Which columns are most important?",
                key="chat_input"
            )
        
        with col2:
            send_button = st.button("Send 📤")

        # Suggested questions
        st.markdown("**💡 Suggested Questions:**")
        suggestions = [
            "What are the key insights from this data?",
            "Which columns have the strongest correlations?",
            "Are there any data quality issues I should know about?",
            "What visualizations would be most helpful?",
            "Which machine learning approach would work best?"
        ]
        
        cols = st.columns(len(suggestions))
        for i, suggestion in enumerate(suggestions):
            with cols[i]:
                if st.button(suggestion, key=f"suggestion_{i}"):
                    user_input = suggestion
                    send_button = True

        # Process chat input
        if (send_button or user_input) and user_input.strip():
            if st.session_state.chatbot:
                with st.spinner("🤖 Thinking..."):
                    response = st.session_state.chatbot.chat(user_input)
                    st.session_state.chat_history.append((user_input, response))
                    st.rerun()
            else:
                st.error("❌ Please load data first or check your API key")

        # Data Preview
        with st.expander("👀 Data Preview", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
        
        # Quick Stats
        with st.expander("📈 Quick Statistics", expanded=False):
            if len(df.select_dtypes(include=[np.number]).columns) > 0:
                st.write("**Numeric Columns Summary:**")
                st.dataframe(df.describe())
            
            st.write("**Dataset Info:**")
            buffer = io.StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())

    else:
        # Welcome screen
        st.info("👆 Please upload a dataset to start chatting with your data!")
        
        st.header("🌟 What You Can Do")
        features = [
            "💬 **Natural Language Queries** - Ask questions about your data in plain English",
            "🔍 **Smart Analysis** - Get insights, correlations, and patterns automatically",
            "📊 **Visualization Suggestions** - AI recommends the best charts for your data",
            "🤖 **Conversational Interface** - Follow-up questions and context-aware responses",
            "📋 **Data Quality Check** - Identify missing values, outliers, and issues",
            "⚡ **Quick Actions** - One-click data summaries and insight discovery"
        ]
        
        for feature in features:
            st.markdown(f"""
            <div class="feature-card">
                {feature}
            </div>
            """, unsafe_allow_html=True)

        st.header("🚀 Getting Started")
        st.markdown("""
        1. **Upload your dataset** (CSV or Excel files supported)
        2. **Add your OpenAI API key** in the sidebar for AI features
        3. **Start chatting** - Ask questions about your data
        4. **Explore insights** - Use suggested questions or ask your own
        
        **Example questions to try:**
        - "What's the most important information in this dataset?"
        - "Show me the relationships between variables"
        - "Are there any quality issues with my data?"
        - "What machine learning models would work best?"
        """)

if __name__ == "__main__":
    main()