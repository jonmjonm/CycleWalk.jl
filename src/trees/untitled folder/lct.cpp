/**
 * Author: 
 * Description: link-cut Tree. Supports BST-like augmentations. (Can be used in place of HLD).
 * Current implementation supports update value at a node, and query max on a path.
 * For details about the structure, refer to https://en.wikipedia.org/wiki/Link/cut_tree
 * Tested on: http://acm.timus.ru/problem.aspx?num=1553
 * Status: Passes existing fuzz tests (with function names modified).
 */
#include <iostream>
using namespace std;
struct Node { // Splay tree. Root's pp contains tree's parent.
	Node *p = 0, *pp = 0, *c[2];
	int v = 0;
	bool flip = 0;
	Node() {c[0] = c[1] = 0; fix(); }
	Node(int val) { v = val; c[0] = c[1] = 0; fix(); }
	void fix() {
		if (c[0]) c[0]->p = this;
		if (c[1]) c[1]->p = this;
		// (+ update sum of subtree elements etc. if wanted)
	}
	void push_flip() {
		if (!flip) return;
		flip = 0; swap(c[0], c[1]);
		if (c[0]) c[0]->flip ^= 1;
		if (c[1]) c[1]->flip ^= 1;
	}
	int up() { return p ? p->c[1] == this : -1; }
	void rot(int i, int b) {
		int h = i ^ b;
		Node *x = c[i], *y = b == 2 ? x : x->c[h], *z = b ? y : x;
		if ((y->p = p)) p->c[up()] = y;
		c[i] = z->c[i ^ 1];
		if (b < 2) {
			x->c[h] = y->c[h ^ 1];
			z->c[h ^ 1] = b ? x : this;
		}
		y->c[i ^ 1] = b ? this : x;
		fix(); x->fix(); y->fix();
		if (p) p->fix();
		swap(pp, y->pp);
	}
	void splay() { /// Splay this up to the root. Always finishes without flip set.
		for (push_flip(); p; ) {
			if (p->p) p->p->push_flip();
			p->push_flip(); push_flip();
			int c1 = up(), c2 = p->up();
			if (c2 == -1) p->rot(c1, 2);
			else p->p->rot(c2, c1 != c2);
		}
	}
	Node* first() { /// Return the min element of the subtree rooted at this, splayed to the top.
		push_flip();
		return c[0] ? c[0]->first() : (splay(), this);
	}
};

struct LinkCut {
	vector<Node> node;
	LinkCut(int N) : node(N) {
		for (int ii=0; ii<N; ii++)
			node[ii].v = ii;
	}

	void link(int u, int v) { // add an edge (u, v)
		assert(!connected(u, v));
		make_root(&node[u]);
		node[u].pp = &node[v];
	}
	void cut(int u, int v) { // remove an edge (u, v)
		Node *x = &node[u], *top = &node[v];
		make_root(top); x->splay();
		assert(top == (x->pp ?: x->c[0]));
		if (x->pp) x->pp = 0;
		else {
			x->c[0] = top->p = 0;
			x->fix();
		}
	}
	bool connected(int u, int v) { // are u, v in the same tree?
		Node* nu = access(&node[u])->first();
		return nu == access(&node[v])->first();
	}
	void make_root(Node* u) { /// Move u to root of represented tree.
		access(u);
		u->splay();
		if(u->c[0]) {
			u->c[0]->p = 0;
			u->c[0]->flip ^= 1;
			u->c[0]->pp = u;
			u->c[0] = 0;
			u->fix();
		}
	}
	Node* access(Node* u) { /// Move u to root aux tree. Return the root of the root aux tree.
		u->splay();
		while (Node* pp = u->pp) {
			pp->splay(); u->pp = 0;
			if (pp->c[1]) {
				pp->c[1]->p = 0; pp->c[1]->pp = pp; }
			pp->c[1] = u; pp->fix(); u = pp;
		}
		return u;
	}
	void display(int ii){
		cout << "(v,l,r,p,pp,r): (" << node[ii].v << ", ";
		if (node[ii].c[0]==0)
			cout << "-1, ";
		else
			cout << node[ii].c[0]->v << ", ";
		if (node[ii].c[1]==0)
			cout << "-1, ";
		else
			cout << node[ii].c[1]->v << ", ";
		if (node[ii].p==0)
			cout << "-1, ";
		else
			cout << node[ii].p->v << ", ";
		if (node[ii].pp==0)
			cout << "-1, ";
		else
			cout << node[ii].pp->v << ", ";
		cout << node[ii].flip << ")" << endl;
	}
};

int main()
{
    LinkCut* tree = new LinkCut(6);
    tree->link(1,2);
    tree->display(1);
    tree->display(2);
    cout<<"This program will calculate the weight of any mass on the moon\n";

    // double moon_g();

}