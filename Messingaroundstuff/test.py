f = open("List_of_features.txt","r")

for x in f:
    output = x.split(" ")[1].replace(',','')
    output = "\"" + output + "\","
    print(output)
