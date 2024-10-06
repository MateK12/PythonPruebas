class Node():
    def __init__(self, data, link):
        self.data = data
        self.link = link
class LinkedList():
    def __init__(self, nodes):
        self.nodes = nodes
        self.head = nodes[0]
        self.tail = nodes[-1]
    def insert(self,node,position=-1):
        if position ==-1:
            for i in self.nodes:
                if i.