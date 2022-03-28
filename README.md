Project Overview

For data sources, I used two reddit threads based on thoughts of both McDonalds and Target employees. In order to extract and process the data, I used the praw library to scrape the comments from these threads. After scraping these comments, I first stored each reddit thread into its own list. From this project, I hoped to see the differences in the sentiment and opinions of employees who work at these two places, and whether the sentiment consensus for each was positive or negative. Furthermore, I hoped to learn in detail how to perform a sentiment analysis from reddit threads and develop graphs and charts to represent my results in an orderly manner.

Implementation

As stated earlier, I began by scraping the comments from both reddit threads. The major components that I wanted to include in my code included data extraction, cleaning the comments data to remove irrelevant characters and to make the formatting more consistent, performing a sentiment analysis, and using a Markov sentence generator to create sentences. When cleaning the data, I had to consider what aspects I wanted to remove. For example, I decided it would be beneficial to remove emojis, capital letters, and lemmatize the words to make them more consistent with each other. I believed that by doing this it would make my frequency dictionary more accurate as the word formatting would be more consistent with each other and the frequency would count correctly. I also wanted to include other analyses such as which words were in a thread that did not exist in the other, and which words existed in both threads. 

One design decision I chose was that I wanted to see the top 20 most common words, but for both positive and negative sentiments for McDonalds and Target employees, instead of just seeing the 20 most common words in general. I chose to do this because I wanted to see what words were driving the sentiment analysis, and whether those words were actually relevant to the sentiment I was trying to capture. For example, words such as “like” have a positive sentiment value but when scraped from the comments section of a reddit thread that can be largely informal, this word can take on a different meaning that might lie more closely with a neutral sentiment. 

Results

For my text analysis, something interesting I discovered was that the overall sentiment for both threads were slightly more positive than negative, though the Target thread did have slightly more positive sentiment than the McDonalds thread. However, the Target thread also had slightly higher negative sentiment than the McDonalds thread. For both of these sentiment analyses, there was an overwhelming amount of neutral sentiment. Before coding this project, I thought there would be much more negative sentiment than the results had shown, especially for an employee working at McDonalds.

![img1](https://i.imgur.com/tUTJ9RC.png)

Additionally, I developed a bar chart for the 20 most common positive and negative words for McDonalds and Target reddit threads separately. The results are as follows. The most common negative words for the McDonalds reddit thread is “shit,” which showed up almost 60 times, “pay,” which showed up around 55 times, and “bad,” which showed up around 47 times. From these three words alone, two seem to have obviously negative connotations; however, the word “pay” for example could have neutral connotations especially when talking about employment. On the other hand, the most common positive words for the McDonalds reddit thread is “like,” “good,” and “pretty.” Once again, since reddit is largely informal, the word “like” could have other definitions that are unrelated to positive connotation. Moving on to the 20 most common negative words for the Target reddit thread is “leave,” “bad,” and “fuck” and so forth. It seems to me that these words clearly denote a negative connotation without question. Moreover, the most common positive words for this thread are “like,” “best,” and “good”. One again, the word “like” has shown up and occurs much more frequently than other words for both threads. To me, this indicates that in hindsight perhaps I should give this word a neutral value, as the word itself can have many meanings and occurs many times.

![img2](https://i.imgur.com/t4nGmDD.png)
![img3](https://i.imgur.com/q65uqRa.png)
![img4](https://i.imgur.com/9Tw1eAI.png)
![img5](https://i.imgur.com/Pktbr6s.png)

Finally, the results of my Markov text synthesizer are as follows:

For McDonalds:

You mean to tell the manager wouldn't take care of by the griller guy, but everything else during rush hour.
But the work is worth 8?
I kind of animal in the kitchen, but sometimes their being dicks makes it all over me.

For Target:

You can’t take it to 3 years at target for two years.
I'll never work retail again thanks to that place, but it was mostly the people I work at.
I've been working for Target from 1996-2015 and honestly I don’t wanna go have to work hard, it’s work, but any good leader knows that some days are going to school, decent pay for my paycheck and anything else is just great life advice in general imo

I made the Markov text synthesizer create three sentences for both McDonalds and Target. Clearly, the sentences that were generated from the Markov captures much of the described sentiment; however the grammatical structure of the sentences itself could be worked on. Perhaps the grammatical errors are caused by the poor grammar of redditor’s comments itself. 

Reflection

What went well was that I managed to learn a lot about how to code this project and how to collect sentiment data from a text. I also learned how to pull data from reddit threads which seemed very intimidating at first. Furthermore, I was able to develop my own code to reach a conclusion that I am fairly pleased with. What I could have done to improve was to be more on top of my work before this project started. I felt that this assignment has been incredibly difficult for me and I was stuck at every possible stage of the process, spending probably more than 20 working hours combined. I know that this is because this course is vigorous, but aspects such as extracting data from reddit and other minor details took up a lot of my time. I wish I knew lists, tuples, and dictionaries in detail before starting this assignment. However, overall, I definitely think that this project will be useful going forward. I never would have thought that I would be able to code such a project from start to finish, extraction to conclusion. In the future, I would probably use a similar code to run through texts which would make work much more efficient.

