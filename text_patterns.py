
# Prefixing patterns by r'\b' ensures they must be at the beginning of a word. 
# This way 'eat' does not match 'great', and most plurals will be OK.


patterns = { 
    # domains
    'cars': r"\b(automobile|car|Toyota|Ford|Dodge|Jeep|Fiat)",
    'animals': r"\b(animal|cat|dog|pet)",
    'food': r"\b(food|diet|eat|restaurant)",
    'programming_language': r"\bprogramming language",
    'sports': r'\baseball(?!: bat)|basketball|badminton|tennis(?!: shoe)|soccer|futbol|football (?!: stadium)',
    'science': r"(?<![:?])(?<!(computer|political|data)) science (?!fiction)|biology|(?<!blood )chemistry|physics|astromony",
    'science_fiction': r"science fiction",
    'data_science': r"data science|machine learning|statistic|quantitative|probability|artificial intelligence|AI\b",
    'information_technology': r"technology|software|information|open source|spark",
    'books': r"\b(book(?!: a )|books|novel|literature|literary)\b",  # 'book' but not 'book a flight'

    # tasks
    'identify': r'identify|classify|which',
    'how_does': r'how (do|does)\b',
    'why_is': r'why (is|are)',
    'what is': r'what (is|are)',
    'extract': r'extract ([^:.?]+) from',
    
    # syntactic structure
    'of_the_following': r'of the following',
    'given_text': r"\b(given a|given the|the given|based on|reference text|as a reference|the following text)",  # passage|paragraph
    'alternatives': r'(,|\b(or)\b)', # This pattern was discovered by POS n-gram clustering

    # output format
    'output_format': r"(display|the results|as a (bulleted )?list|list them|format this|format them|in the format|each record|new line|comma separated|separated by|separate them with a comma|JSON|XML|markdown)", 
}