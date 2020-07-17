from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup as soup

my_url = 'https://www.newegg.com/Video-Cards-Video-Devices/Category/ID-38?Tpk=graphics%20cards'

uClient = uReq(my_url)

# opening up connection, grabbing the page
page_html = uClient.read()
uClient.close()

# html parsing
page_soup = soup(page_html, "html.parser")

# print(page_soup.h1)
# print(page_soup.p)

# grabs each product
containers = page_soup.find_all("div", {"class": "item-container"})

filename = "gpu_products.csv"
f = open(filename, "w")
headers = "brand, product_name, shipping\n"
f.write(headers)

for container in containers:
    brand = container.findAll("a", {"class": "item-brand"})
    brand = brand[0].img["title"]
    title_container = container.findAll("a", {"class": "item-title"})
    product_name = title_container[0].text

    shipping_container = container.findAll("li", {"class": "price-ship"})
    shipping = shipping_container[0].text.strip()

    print("\nbrand: " + str(brand))
    print("product name: " + product_name)
    print("shipping: " + shipping)

    f.write(brand + "," + product_name.replace(",", "|") + "," + shipping + "\n")

f.close()
