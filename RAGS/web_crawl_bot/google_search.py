
from serpapi import GoogleSearch
import gradio as gr

"""
results.keys()
dict_keys(['search_metadata', 'search_parameters', 'search_information', 'knowledge_graph', 'inline_images', 'top_stories', 'organic_results', 'related_searches', 'pagination', 'serpapi_pagination'])

My api_key= "80b064ee5dd971935cdf6c205ea1620534456d660654a6066e6e286e36577620"
"""

class MyGoogleSearch :

  def __init__(self, api_key="80b064ee5dd971935cdf6c205ea1620534456d660654a6066e6e286e36577620") :
    self.engine = "google"
    self.api_key = api_key
    self.results= {}
    self.related_results= []
    self.result_key ='organic_results'

  def CoreSearch(self, query):
    search = GoogleSearch({
      "engine" : self.engine,
      "q": query,
      "api_key": self.api_key,
    })
    results = search.get_dict()
    return results

  def RelatedSearch(self, count=7):
      for related in self.results['related_searches'][0:count]:
          rel_results= self.CoreSearch(related['query'])
          self.related_results.append(rel_results)
  
  def Search(self, query, related_search=False, related_depth=1,
             rec_search=False, rec_depth=1):
      self.results = self.CoreSearch(query)
      if related_search:
          self.RelatedSearch()
      return self.results

  def _GetLinks(self, results, result_links):
      for result in results[self.result_key][0:10]:
          if result['title'] in result_links :
              continue;
          result_links[result['title']] = result['link'] 

  def GetLinks(self):
    result_links = {}
    self._GetLinks(self.results, result_links)
    for related in self.related_results:
      self._GetLinks(related, result_links)
    return result_links

  def show_links(self, count=33):
      links = self.GetLinks()
      print("================================================================")
      for i, (title,link) in enumerate(links.items()):
          print(i, title, "\t: ", link)
      print("================================================================")


  def get_result_list(self, count=12):
    res = []
    count = len(self.results) if len(self.results) < count else count
    for ind, result in enumerate(self.results[self.result_key][0:count]):
        res.append(str(result['link']))
    return res

  def top_10(self):
    ret = ""
    for ind, result in enumerate(self.results[self.result_key][0:10]):
      ret = ret + str(ind) + ". " + str(result['link']) + "\n"
    return ret

  def show_results(self, count=33):
    print("----------------------------------------")
    for result in self.results[self.result_key][0:count]:
      print(result['position'], result['title'])
      print(result['link'])
      print()
    print("----------------------------------------")



#Google = MyGoogleSearch()
#results = Google.Search("Intel CPU Performance Data", True)
#Google.show_results()
#Google.show_links()


def google_it(query, history):
    google = MyGoogleSearch()
    google.Search(query)
    #return google.top_10()
    return google.get_result_list()

if __name__ == '__main__' :
  gr.ChatInterface(google_it).launch(share=True)
