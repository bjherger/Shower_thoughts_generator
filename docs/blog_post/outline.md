# Outline

Intro
 
 - Example (gif?)
 - Purpose
 
 
Data

 - Intro to Reddit / shower thoughts
 - Quick discussion of PRAW
 - Table with fields parsed
 
 - Discussing of parsing
   - Character filtering / formatting
   - Start and end tokens
   - Window creation
   
Model
 
 - Given set of characters, predict the next character
 - Largely based on Keras example
 - Modified Keras to allow for better generalization, out of sample testing
 - Added character embedding
   - Would be interesting to look at clustering
 - Word model would likely be better
 - Model seems to get stuck in local minima, would be interesting to look at restarts
 - Would also be interesting to predict multiple characters at a time
 
Conclusion
 
 - Weak character level model
 - Strong ability to learn words, basic grammar
 - Super expensive train time
 - Check out repo
