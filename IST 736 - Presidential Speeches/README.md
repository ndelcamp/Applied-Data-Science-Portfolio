# IST 736 - Presidential Speeches
Collaborative project with Drew Howell where we scraped and analyzed presidential speeches, studying sentiment over time and the ability to predict certain features of each president and the time period in which they gave their speeches.

Files in this folder:
  - Final_Report.docx - Report of methods and findings. Review with Microsoft Word.
  - Final_Presentation.pptx - Presentation of methods and findings. Review with Microsoft PowerPoint
  - Scrape_Data.py - Python program that scrapes the presidential speeches and demographic data off of The Miller Center. Review with any text editor.
  - EDA.py - Python program performing exploratory data analysis. Review with any text editor.
  - TFIDF_Dist.py - Python program that vectorizes the presidential speeches using TFIDF vectorization and creates visuals to represent the vectorization. Review with any text editor.
  - Models.py - Python program that creates, evaluates, and produces visualizations of differnent machine learning models to cluster and model demographic information and time period of speeches using only the content of the speeches.
  - presidents.csv - Output from Scrape_Data.py. Contains information about each president. Gaps were filled in other programs. Review with any text editor or Microsoft Excel.
  - speeches - Folder containing 122 speeches in .txt format. These are the raw speeches which were cleaned in TFIDF_Dist.py. Review these files with any text editor.
  
