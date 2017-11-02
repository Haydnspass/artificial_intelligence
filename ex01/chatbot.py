#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Artificial Intelligence
Prof. Björn Ommer
WS17/18

Exercise sheet 1 - Question 1

@author: Uta Büchler
"""
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import sys

random.seed()

class Chatbot():
    
    def __init__(self,greetings,questions,reaction,farewells,questions_user=None,answers=None):
        '''
        Initialize the chatbot given the input arguments
        (is called when creating an object of the class Chatbot)
        
        input:  greetings:      list of strings - predefines which greetings the bot can use
                questions:      list of strings - predefines the questions the bot can ask
                reaction:       function - decides how to react to the answers made by the user
                questions_user: list of strings - predefines which questions the user might ask the bot (Ex. 1b)
                answers:        list of strings - predefines the answers of the bot given a specific question of the user
                farewells:      list of strings - predefines how the bot can say goodbye
        
        '''
        self.reaction = reaction #function
        self.greetings = greetings
        self.questions = questions
        self.questions_user = questions_user
        self.answers = answers
        self.farewells = farewells
        #during the conversation the bot has to ask all questions
        #question index is removed in askQuestion() if question was asked
        self.notAsked = list(range(0,len(questions)))
        self.shownGauss = 0 #boolean - has to be changed to 0 for Question 1c

    
    def greeting(self):
        '''
        Question 1a)
        The bot outputs randomly a greeting given all possible greetings in self.greetings
        Afterwards, the bot asks for the name of the user and reacts by saying 'Nice to meet you USER',
        where USER has to be replaced by the input made from the user.

        Question 1b)
        After greeting the user the Bot has to check if the user asked questions during his last input.
        If yes, he has to answer them. For extracting the different parts of the users input (in case
        he asked a question) you can use self.extractParts(user_input,delimiter)
        '''

        selectedGreeting = self.greetings[random.randint(0,self.greetings.__len__()-1)]
        print(selectedGreeting)
        answeredName = raw_input("Whats ur name?")
        print('Nice to meet u!')
        if self.isQuestion(user_input=answeredName):
            splittedAnswer = self.extractParts(user_input=answeredName,delimiter='.')
            self.answerQuestion(user_input=splittedAnswer[1])

        
    def askQuestion(self):
        '''
        Question 1a)
        The bot outputs randomly a question given all possible questions in self.questions
        If the question was asked, it should be removed from the variable self.notAsked.
        Afterwards, the bot should react to the answer of the user by using the input of the user.
        Example: 
            Question_bot: What are your hobbies? 
            Answer_user: football, meeting friends
            Reaction_Bot: I don't like football, but I like meeting friends
            
        How the bot reacts to which question has to be given by the function self.reaction
        For extracting the different parts of the answer (1st part is 'football', second is 'meeting friends')
        you can use the function self.extractParts()
        
        Question 1b)
        After reacting to the answer of the user the Bot has to check if the user asked questions during his last input.
        If yes, he has to answer them. For extracting the different parts of the users input (in case
        he asked a question) you can use self.extractParts(user_input,delimiter)
        '''
        selectedQuestion = random.choice(self.notAsked)
        self.notAsked.remove(selectedQuestion)
        answer = raw_input(self.questions[selectedQuestion])
        if self.isQuestion(user_input=answer):
            splittedAnswer = self.extractParts(user_input=answer, delimiter='.')
            reaction(self, self.questions[selectedQuestion], splittedAnswer[0])
            self.answerQuestion(user_input=splittedAnswer[1])
        else:
            reaction(self, self.questions[selectedQuestion], answer)

        
    def farewell(self):
        '''
        Question 1a)
        The bot outputs randomly a farewell given all possible greetings in self.farewells
        '''
        print(random.choice(self.farewells))
                                 
    def extractParts(self,user_input,delimiter,outputType = str):
        '''
        Find the different parts of the users input, by splitting the string 'user_input' in its parts
        using the delimiter specified as input. If necessary, the output can be of a different type than
        string (for example float)
        
        Input:
            user_input: input made by the user
            delimiter:  substring which identifies the end and start of a part
            outputType: type of the elements in the output list 'parts'
        
        Output:
            parts: list of strings - contains the parts of the user's input in the sepecified type
        '''
        parts = [outputType(x.strip()) for x in user_input.split(delimiter)]
        
        return parts
    
    def answerQuestion(self,user_input):
        '''
        Question 1b)
        This function finds the appropriate answer to the question(s) made by the user.
        First, the question has to be find in self.questions_user. This then leads to the index
        needed to find the right answer of the bot using self.answers.
        
        Input: user_input: list of strings - contains the parts of the user's input
        Output: answers: all answers to the questions made from the user in his/her current input
    
        '''

        questionIsAnswered = False
        for j in range(len(self.questions_user)):
            if user_input == self.questions_user[j]:
                print(self.answers[j])
                questionIsAnswered = True
        if not questionIsAnswered:
            print('Err: Uncaught question.')


    
    
    
    def isQuestion(self,user_input):
        '''
        Question 1b)
        This function checks if the user asked a question during his last input.
        Input:  user_input: list of strings - contains the parts of the user's input
        Output: a boolean variable, stating if one of the user_input parts contains a questionmark or not
        '''

        if user_input[-1] == '?':
            return True
        else:
            return False

    def input():
        pass
    

    def plotGauss(self):
        '''
        Question 1c)
        This function let the chatbot plot two Gauss Curves given a mean and two different standard deviations
        chosen by the user.
        
        Use the function "norm" from the library scipy.stats to compute the function values y1 and y2
        given x, the mean and the two standard deviations std1 and std2
        Plot both functions (as subplots) using the library matplotlib.pyplot.
        Play a little bit around with the properties one can set for the plots
        (like setting a title,changing the line color etc etc)
        '''
        self.shownGauss=1
        user_input=raw_input("Do you want to see how a Gauss curve looks like? \n")
        if user_input.lower()=='yes':
            user_input = raw_input("Cool! Then you need to give me a random number between 0 and 1 for the mean \n")
            mean = float(user_input)
            user_input = raw_input("I want to show you 2 Gaussians with different standard deviations. So I need two numbers now between 0 and 1 \n")
            stds = self.extractParts(user_input,',',outputType=float)
            
            x = np.linspace(-3,3,1000)
            for i in range(stds.__len__()):
                y = norm.pdf(x, loc = mean, scale = stds[i])
                plt.plot(x,y)
                plt.show()
            
        elif user_input.lower()=='no':
            print("Oh... ok.")
            
    def main(self):
        '''
        main function to perform the conversation between chatbot and user
        starts with greeting, then asks all available questions and ends with a farewell
        
        The part with the Gauss curve only applies for Question 1c.
        Change the boolean variable self.shownGauss to 0, so that the function plotGauss will be executed
        during the conversation.
        '''

        self.greeting()
        while not len(self.notAsked) == 0:#ask questions until all questions are asked
            self.askQuestion()
            if not self.shownGauss:#plot Gauss if it hasn't been done already
                prob = 1 if len(self.notAsked)==0 else random.randint(0,1)#plot with probability of 50% or 100% if last question was asked
                if prob: self.plotGauss()


        self.farewell()








if __name__ == "__main__":
    
    def reaction(bot,question,answer):
        '''
        Question 1a)
        Determines the reaction of an instance of a chatbot to an answer given by the user after the bot asked a question
        Input:
            bot:        Object of the class Chatbot
            question:   string - containing the question the bot asked
            answer:     string - containing the answer of the user to the question of the bot
        Output: nothing - the output/reaction should be printed in the console
        '''
        if not question.find('Trump') == -1:
            print(answer + ', as well.')
        if not question.find('hobbies') == -1:
            partsAnswer = bot.extractParts(user_input=answer, delimiter=',')
            print('Oh, that is so cool. I like ' +
                  ' '.join([partsAnswer[i] for i in range(partsAnswer.__len__())]) + ' as well')
        if not question.find('instrument') == -1:
            print('Cool. Listening to music is great.')


    # To use the python 3 function input while pretending to use raw_input
    raw_input = input
    
    bot = Chatbot(greetings = ['Hey dude!','Hi boy!', 'Hello!'],
              questions = ['What r ur hobbies?','What instrument do u play?',
                           'What do u think of Donald Trump? (start with I think)'],
              reaction = reaction,
              questions_user = ['What is ur name?','Is this a private conversation?',
                                'Will the exercises be updated to Python 3?'],
              answers =  ['My name is Bob the bot.','I hope so, but do not send me nudes man.','I do not think, but I hope so.'],
              farewells = ['See u','Bye','Bb'])
    bot.main()


