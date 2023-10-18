import os
import random
import re
import sys
import numpy

DAMPING = 0.85
SAMPLES = 10000
TOLERANCE = 1e-3  # Error tolerance = ±0.001 when comparing sample and iterate results


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    n = len(corpus[page])  # number of out going links in page
    N = len(corpus)  # number of pages in corpus
    p = (1-damping_factor)/N  # probability to choose any single page
    # transiton model for page
    # all pages have same probability p to be chosen
    tm = {k: p for k in corpus.keys()}

    for k, v in corpus.items():
        if k == page:
            if not v:
                for pp in corpus.keys():
                    tm[pp] = 1/N
            else:
                for link in v:
                    # we add the probability of choosing link from k page
                    tm[link] += damping_factor/n
    return tm


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pr = dict()
    samples = []
    starting_page = random.choice(list(corpus.keys()))
    page = starting_page

    for i in range(n):  # compute n samples
        tm = transition_model(corpus, page, damping_factor)
        # returns a random page from list of keys according to
        # probability distribution of transition model for page
        kys = list(tm.keys())
        vals = list(tm.values())
        next_page = random.choices(kys, vals, k=1).pop()
        samples.append(next_page)
        page = next_page

    for k in corpus:
        n_k = samples.count(k)
        pr[k] = n_k/n

    return pr


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    N = len(corpus)  # number of pages in corpus
    p = (1-damping_factor)/N  # probability to choose any single page
    # transiton model for page
    # initial pr value : all pages have same probability to be chosen
    initial = {k: 1/N for k in corpus.keys()}
    next = calculate_pr(corpus, damping_factor, initial, N)

    while has_not_converged(initial, next):
        initial = next
        # calculate new pr values
        next = calculate_pr(corpus, damping_factor, initial, N)
    return next


def has_not_converged(previous, current):
    i = random.choice(list(previous.keys()))
    return True if numpy.abs(current[i]-previous[i]) > TOLERANCE else False


def calculate_pr(corpus, damping_factor, current, N):
    new = {k: (1-damping_factor)/N for k in corpus.keys()}
    for page in corpus.keys():
        for key, links in corpus.items():
            if not links:  # key page with no link. We can pretend key has a link to all pages including itself.
                new[page] += damping_factor * current[key] / N
            if page in links:
                new[page] += damping_factor * current[key] / len(links)
    return new


if __name__ == "__main__":
    main()
