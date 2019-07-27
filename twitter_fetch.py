import tweepy
import math
import string

APP_KEY = '***'
APP_SECRET = '***'

auth = tweepy.OAuthHandler('***', '***')
auth.set_access_token(APP_KEY, APP_SECRET)

api = tweepy.API(auth)

tweets = api.search(q = "***", count = 100, result_type = "recent")

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
    self.tweettoid = []
    tweetto_obj = []
    temp_obj = None
    for i in range(len(self.text)):
        try:
            temp_tweetto = self._json["text"].split(' ')
        except AttributeError:
            temp_tweetto = 'other'

    for i in range(len(temp_tweetto)):
        temp_tweetto[i] = temp_tweetto[i].rstrip(string.punctuation)
        try:
            if temp_tweetto[i][0] == '@':
                self.tweetto.append(temp_tweetto[i][1:])
            else:
                continue
        except IndexError:
            continue

    for i in range(len(self.tweetto)):
        try:
            self.tweettoid.append(api.get_user(screen_name=self.tweetto[i]))
        except tweepy.TweepError:
            # print('{} cannot be found.'.format(self.tweetto[i]))
            continue
        # print(temp_obj.id)

for tweet in tweets:
    tweetto(tweet)
    # print('{} - {}'.format(tweet.user.name, tweet.tweetto))

# print('=== Sender Locations ===')
# for tweet in tweets:
#     print(tweet.user.location)
# print('=== Receiver Locations ===')
# for i in range(len(tweets)):
#     for j in range(len(tweets[i].tweettoid)):
#         print(tweets[i].tweettoid[j].location)

'''
Find User Location Coor
'''

def set_user_coor(self):
    self.lat = None
    self.long = None

    if 'Sydney' in self.location:
        self.lat = -33.865
        self.long = 51.210
    if 'Albury' in self.location:
        self.lat = -36.081
        self.long = 146.916
    elif 'Melbourne' in self.location:
        self.lat = -37.814
        self.long = 144.963
    elif 'Geelong' in self.location:
        self.lat = -38.147
        self.long = 144.361
    elif 'Darwin' in self.location:
        self.lat = -12.463
        self.long = 130.842
    elif 'Perth' in self.location:
        self.lat = -31.954
        self.long = 115.857

def get_dist(self):
    self.dist_to_receivers = []
    for receiver in self.tweettoid:
        try:
            self.dist_to_receivers.append(math.sqrt((self.user.lat - receiver.lat)**2 + (self.user.long - receiver.long)**2))
        except TypeError:
            # self.dist_to_receivers.append(None)
            continue

def get_dist_distro(tweets):
    distance_distribution = []
    for tweet in tweets:
        for num in tweet.dist_to_receivers:
            distance_distribution.append(num)
    return distance_distribution

for tweet in tweets:
    set_user_coor(tweet.user)
    # print('{}, {}'.format(tweet.user.lat, tweet.user.long))
    for receiver in tweet.tweettoid:
        set_user_coor(receiver)
        # print('{}, {}'.format(receiver.lat, receiver.long))

for tweet in tweets:
    get_dist(tweet)
    # print(tweet.dist_to_receivers)

distance_distribution = get_dist_distro(tweets)
# print(distance_distribution)
