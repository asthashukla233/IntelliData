# 💬 IntelliData

An AI-powered conversational data analysis tool built with Streamlit that lets you interact with your datasets using natural language queries.


## 🌟 Features

### 🤖 AI-Powered Analysis
- **Natural Language Queries**: Ask questions about your data in plain English
- **Intelligent Insights**: Get automated analysis, correlations, and pattern detection
- **Conversational Interface**: Context-aware responses with follow-up question support
- **Smart Suggestions**: AI recommends visualizations and analysis approaches

### 📊 Data Processing
- **Multi-format Support**: CSV, Excel (XLSX, XLS) file upload
- **Automatic Column Detection**: Identifies numeric, categorical, and datetime columns
- **Data Quality Assessment**: Missing values, duplicates, and outlier detection
- **Advanced Preprocessing**: Handles various encodings and data formats

### 🔍 Analysis Capabilities
- **Statistical Summaries**: Comprehensive dataset overviews
- **Correlation Analysis**: Identify relationships between variables
- **Missing Data Analysis**: Detailed missing value patterns
- **Data Profiling**: Memory usage, data types, and quality metrics

### 🎨 Interactive Interface
- **Modern UI**: Gradient backgrounds and responsive design
- **Real-time Chat**: Instant responses with conversation history
- **Quick Actions**: One-click data summaries and insights
- **Data Preview**: Interactive dataframe viewing

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
OpenAI API Key (for AI features)
```

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd chat-with-your-data
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run streamlit_app.py
```

4. **Open your browser**
Navigate to `http://localhost:8501`

## 📋 Requirements

Create a `requirements.txt` file with:

```txt
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.21.0
plotly>=5.15.0
seaborn>=0.11.0
matplotlib>=3.5.0
scikit-learn>=1.3.0
openai>=1.0.0
```

## 🔧 Configuration

### OpenAI API Setup

1. **Get your API key**
   - Visit [OpenAI Platform](https://platform.openai.com/api-keys)
   - Create a new API key
   - Copy the key (starts with `sk-`)

2. **Add to the application**
   - Enter your API key in the sidebar
   - The app will validate and activate AI features

### Environment Variables (Optional)

```bash
# Create .env file
OPENAI_API_KEY=your_api_key_here
```

## 📖 Usage Guide

### 1. Upload Your Data
- Click "Upload your dataset" in the sidebar
- Supported formats: CSV, XLSX, XLS
- Click "🚀 Load Data" to process

### 2. Configure AI (Optional)
- Add your OpenAI API key in the sidebar
- Status indicator shows connection status
- AI features activate automatically

### 3. Start Chatting
- Use the chat input at the bottom
- Try suggested questions or ask your own
- View conversation history in the chat container

### 4. Explore Features
- **Quick Actions**: Data summary, insights discovery
- **Data Preview**: View your dataset
- **Quick Statistics**: Numerical summaries

## 💡 Example Questions

### Data Overview
- "What's the most important information in this dataset?"
- "Give me a comprehensive summary of this data"
- "How many rows and columns do I have?"

### Analysis & Insights
- "Which columns have the strongest correlations?"
- "What patterns do you see in the data?"
- "Are there any outliers I should know about?"

### Data Quality
- "Are there any data quality issues?"
- "Which columns have missing values?"
- "How should I handle the missing data?"

### Visualization
- "What visualizations would be most helpful?"
- "How should I plot the relationship between X and Y?"
- "What's the best way to show this data?"

### Machine Learning
- "What machine learning approach would work best?"
- "Which features are most important?"
- "How should I prepare this data for modeling?"

## 🏗️ Architecture

### Core Components

```
streamlit_app.py
├── DataChatbot Class
│   ├── OpenAI Integration
│   ├── Conversation Management
│   ├── Context Generation
│   └── Fallback Analysis
├── AdvancedDataProcessor Class
│   ├── File Loading
│   ├── Column Analysis
│   ├── Data Profiling
│   └── Type Detection
└── Streamlit Interface
    ├── Chat Interface
    ├── Data Upload
    ├── Configuration
    └── Visualization
```

### Key Features

- **Error Handling**: Robust error handling for API calls and data processing
- **Fallback Mode**: Basic analysis without API key
- **Session Management**: Maintains conversation context
- **Auto-detection**: Smart column type identification

## 🛠️ Customization

### Adding New Analysis Functions

```python
def custom_analysis(self, df, user_query):
    """Add your custom analysis logic"""
    # Your analysis code here
    return analysis_result
```

### Extending Visualization Options

```python
def create_custom_viz(df, viz_type):
    """Add custom visualization types"""
    # Your visualization code here
    return plotly_figure
```

### Custom Chat Responses

```python
def _custom_fallback(self, user_message):
    """Add custom fallback responses"""
    # Your logic here
    return response
```

## 🔒 Security & Privacy

### Data Handling
- **Local Processing**: All data processing happens locally
- **No Data Storage**: Data is not permanently stored
- **Session-based**: Data cleared when session ends

### API Security
- **Client-side Only**: API keys stored in session state
- **No Logging**: API keys not logged or stored
- **Secure Transmission**: HTTPS communication with OpenAI

## 🐛 Troubleshooting

### Common Issues

**API Key Authentication Failed**
- Verify API key starts with `sk-`
- Check OpenAI account billing status
- Ensure sufficient API credits

**File Upload Issues**
- Check file format (CSV, XLSX, XLS)
- Try different encoding if CSV fails
- Ensure file is not corrupted

**Performance Issues**
- Large datasets may take longer to process
- Consider sampling for very large files
- Close other browser tabs if needed

### Error Messages

| Error | Solution |
|-------|----------|
| "Invalid API Key Format" | Ensure key starts with `sk-` |
| "Rate limit exceeded" | Wait before next request |
| "Could not decode CSV" | Try saving CSV with UTF-8 encoding |
| "OpenAI package not installed" | Run `pip install openai` |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Streamlit** for the amazing web app framework
- **OpenAI** for GPT-3.5 integration
- **Plotly** for interactive visualizations
- **Pandas** for data manipulation

## 📞 Support

- **Issues**: Open a GitHub issue
- **Documentation**: Check the code comments
- **API Help**: Visit [OpenAI Documentation](https://platform.openai.com/docs)

## 🗺️ Roadmap

### Upcoming Features
- [ ] Support for more file formats (JSON, Parquet)
- [ ] Advanced visualization builder
- [ ] Data export functionality
- [ ] Multi-dataset comparison
- [ ] Automated report generation
- [ ] Integration with other AI models

### Version History
- **v1.0.0**: Initial release with core chat functionality
- **v1.1.0**: Enhanced error handling and UI improvements
- **v1.2.0**: Advanced data profiling and analysis

---

Made with ❤️ using Streamlit and OpenAI
