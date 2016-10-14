library(dplyr)
library(readr)
library(ggplot2)

root_folder="/home/hieuvq/Documents/Competitions/Outbrain"

## Promoted content --------------------------------------------------------
promoted.content <- read_csv(paste(root_folder,"/input/promoted_content.csv",sep=''))

# no. of advertiser
attach(promoted.content)
  n.advertiser <- length(unique(advertiser_id))
  n.campaign <- length(unique(campaign_id))
  n.doc.with.promote <- length(unique(document_id))
detach(promoted.content)
  

## Doc categs --------------------------------------------------------------
doc.categs <- read_csv(paste(root_folder,"input/documents_categories.csv"))

attach(doc.categs)
  n.doc <- length(unique(document_id))
  n.categ <- length(unique(category_id))
  categ.per.doc <- group_by(doc.categs, document_id) %>% summarise(n.categ = length(category_id)) %>% arrange(desc(n.categ))
detach(doc.categs)  
  
## Usually, how many categs per doc?
attach(categ.per.doc)
  hp <- ggplot(categ.per.doc, aes(x = n.categ)) + geom_bar(width = .1) + scale_x_discrete(limits = 1:2)
  hp <- hp + xlab("# categs") + ylab("# documents")
  
