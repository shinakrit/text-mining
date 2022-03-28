Project Overview
For data sources, I used two reddit threads based on thoughts of both McDonalds and Target employees. In order to extract and process the data, I used the praw library to scrape the comments from these threads. After scraping these comments, I first stored each reddit thread into its own list. From this project, I hoped to see the differences in the sentiment and opinions of employees who work at these two places, and whether the sentiment consensus for each was positive or negative. Furthermore, I hoped to learn in detail how to perform a sentiment analysis from reddit threads and develop graphs and charts to represent my results in an orderly manner.

Implementation
As stated earlier, I began by scraping the comments from both reddit threads. The major components that I wanted to include in my code included data extraction, cleaning the comments data to remove irrelevant characters and to make the formatting more consistent, performing a sentiment analysis, and using a Markov sentence generator to create sentences. When cleaning the data, I had to consider what aspects I wanted to remove. For example, I decided it would be beneficial to remove emojis, capital letters, and lemmatize the words to make them more consistent with each other. I believed that by doing this it would make my frequency dictionary more accurate as the word formatting would be more consistent with each other and the frequency would count correctly. I also wanted to include other analyses such as which words were in a thread that did not exist in the other, and which words existed in both threads. 

One design decision I chose was that I wanted to see the top 20 most common words, but for both positive and negative sentiments for McDonalds and Target employees, instead of just seeing the 20 most common words in general. I chose to do this because I wanted to see what words were driving the sentiment analysis, and whether those words were actually relevant to the sentiment I was trying to capture. For example, words such as “like” have a positive sentiment value but when scraped from the comments section of a reddit thread that can be largely informal, this word can take on a different meaning that might lie more closely with a neutral sentiment. 

Results
For my text analysis, something interesting I discovered was that the overall sentiment for both threads were slightly more positive than negative, though the Target thread did have slightly more positive sentiment than the McDonalds thread. However, the Target thread also had slightly higher negative sentiment than the McDonalds thread. For both of these sentiment analyses, there was an overwhelming amount of neutral sentiment. Before coding this project, I thought there would be much more negative sentiment than the results had shown, especially for an employee working at McDonalds.

![img1](https://i.imgur.com/tUTJ9RC.png)
![img2](https://i.imgur.com/t4nGmDD.png)
![img3](https://i.imgur.com/q65uqRa.png)
![img4](https://i.imgur.com/9Tw1eAI.png)
![img5](https://i.imgur.com/Pktbr6s.png)
