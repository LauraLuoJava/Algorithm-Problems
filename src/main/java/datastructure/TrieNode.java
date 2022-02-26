package datastructure;

class TrieNode{
    boolean isEnd = false;
    TrieNode[] child = new TrieNode[26];
}

class Trie {

    TrieNode root;

    public Trie() {
        root = new TrieNode();
    }

    public void insert(String word) {
        TrieNode node = root;
        for (char c : word.toCharArray()){
            if (node.child[c-'a'] == null)
                node.child[c-'a'] = new TrieNode();
            node = node.child[c-'a'];
        }
        node.isEnd = true;
    }

    public boolean search(String word) {
        TrieNode node = root;
        for (char c : word.toCharArray()){
            if (node.child[c-'a'] == null)
                return false;
            node = node.child[c-'a'];
        }
        return node.isEnd;
    }

    public boolean startsWith(String prefix) {
        TrieNode node = root;
        for (char c : prefix.toCharArray()){
            if (node.child[c-'a'] == null)
                return false;
            node = node.child[c-'a'];
        }
        return true;
    }
}
class WordDictionary {
    TrieNode root;
    public WordDictionary() {
        root = new TrieNode();
    }

    public void addWord(String word) {
        TrieNode node = root;
        for (char c : word.toCharArray()){
            if (node.child[c-'a'] == null)
                node.child[c-'a'] = new TrieNode();
            node = node.child[c-'a'];
        }
        node.isEnd = true;

    }

    public boolean search(String word) {
        return searchRecursion(word,root);
    }

    public boolean searchRecursion(String word, TrieNode root){
        TrieNode node = root;
        for (int i = 0; i < word.length(); i++) {
            char c = word.charAt(i);
            if (c == '.') {
                for (TrieNode cNode : node.child) {
                    if (cNode != null && searchRecursion(word.substring(i + 1), cNode))
                        return true;
                }
                return false;
            } else if (node.child[c - 'a'] == null)
                return false;
            node = node.child[c - 'a'];
        }
        return node.isEnd;
    }
}
