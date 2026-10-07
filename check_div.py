from html.parser import HTMLParser
c = 0
class P(HTMLParser):
    def handle_starttag(self, tag, attrs):
        global c
        if tag == 'div': c += 1
    def handle_endtag(self, tag):
        global c
        if tag == 'div': c -= 1
p = P()
with open('c:\\Users\\User\\Downloads\\bopis\\bopis.html', encoding='utf-8') as f:
    p.feed(f.read())
print("Unclosed div count:", c)
