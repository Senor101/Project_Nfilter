sfrom selenium import webdriver
import os
import urllib.request
import time

path = r'C:\Program Files (x86)\chromedriver.exe'

url_prefix = "https://www.google.com.sg/search?q="
url_postfix = "&source=lnms&tbm=isch&sa=X&ei=0eZEVbj3IJG5uATalICQAQ&ved=0CAcQ_AUoAQ&biw=939&bih=591"

save_folder = 'train'

def main():
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    download_images()
    
def download_images():
    #no of images you want
    n=100;

    #keyword for what you are searching 
    search_for='null'
    
    search_url = url_prefix+search_for+url_postfix
    
    path = r'C:\Program Files (x86)\chromedriver.exe'
    
    driver = webdriver.Chrome(path)
    driver.get(search_url)
    
    value = 0
    for i in range(3):
        driver.execute_script("scrollBy("+ str(value) +",+1000);")
        value += 1000
        time.sleep(1)
    
    elem1 = driver.find_element_by_id('islmp')
    sub = elem1.find_elements_by_tag_name('img')
    
    count=0
    for j,i in enumerate(sub):
        if j < n:
            src = i.get_attribute('src')                         
            try:
                if src != None:
                    src  = str(src)
                    
                    urllib.request.urlretrieve(src, os.path.join(save_folder, search_for+str(count)+'.jpg'))
                else:
                    raise TypeError
            except Exception as e:              #catches type error along with other errors
                print(f'fail with error {e}')

            count+=1
    
    driver.close()
    
if __name__ == "__main__":
    main()
    
