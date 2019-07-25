import tweepy

APP_KEY = '2992206428-EWTAaNtejE8vyBblrWGsu2CmzFWIbqLEr6P0ur1'
APP_SECRET = '78SaivREPdXxEkCz2EATcQ1XWqCJdiGCffIukFmQDGblj'

auth = tweepy.OAuthHandler('6dK8YxPU6sksol1bIFgzoJ3JN', 'CQTfLmp0orR6ByYzoyYE6CqqnlKIEkNEcGsHBKsiqjkYpwa3Hd')
auth.set_access_token(APP_KEY, APP_SECRET)

api = tweepy.API(auth)

tweets = api.search(q = "#sydneytrains", count = 100, result_type = "recent")

# print('=== Status Object ===')
# print('\n')
# print(tweets[0].__dict__.keys())
# print('=== User Object ===')
# print('\n')
# print(tweets[0].user.__dict__.keys())
# print('\n')
# print('=== User ID ===')
# for tweet in tweets:
#     print(tweet.user.id)
'''
Count User frequency using this hastag
'''
def user_list(tweets):
    id_ls = []
    for tweet in tweets:
        id_ls.append(tweet.user.id)
    return id_ls

def count_freq(my_list):
    # Code from https://www.geeksforgeeks.org/counting-the-frequencies-in-a-list-using-dictionary-in-python/
    # Creating an empty dictionary
    freq = {}
    for item in my_list:
        if (item in freq):
            freq[item] += 1
        else:
            freq[item] = 1

    # for key, value in freq.items():
    #     print ("% d : % d"%(key, value))

user_ls = user_list(tweets)
count_freq(user_ls)

'''
Get user location
'''
def user_loc(tweets):
    my_dict = {}
    for tweet in tweets:
        my_dict[tweet.user.id] = tweet.user.location

    # print(my_dict.keys())
    # print(my_dict.values())

user_loc(tweets)

'''
Find user mentions
'''
def tweetto (self):
    self.tweetto = []
    for i in range(len(self.text)):
        try:
            temp_tweetto = self.text.split(' ')
        except AttributeError:
            temp_tweetto = 'other'

    for i in range(len(temp_tweetto)):
        try:
            if temp_tweetto[i][0] == '@':
                self.tweetto.append(temp_tweetto[i][1:-1])
            else:
                continue
        except IndexError:
            continue

for tweet in tweets:
    tweetto(tweet)
    print('{} - {}'.format(tweet.user.name, tweet.tweetto))
