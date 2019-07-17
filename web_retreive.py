import requests
from lxml import html

htmlsource = requests.get('http://shyjohn.net/blog').text
tree = html.fromstring(htmlsource)

with open('C:/Users/user/Documents/Python Course/test/shy_john.html', mode = 'w') as fo:
    fo.write(htmlsource)

fo.close()

blog_content = tree.xpath('//*/div/div/div[2]/p')

blog_content_text = [r.text.strip() for r in blog_content]
print(blog_content_text)
