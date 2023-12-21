# Propriétés théoriques des GANs

Les réseaux adversoriels génératifs (GAN) sont une classe d'algorithmes génératifs dont il a été démontré qu'ils produisent des échantillons de pointe, en particulier dans le domaine de la création d'images mais aussi de données synthétiques très proches des données réelles. Cette section du notebook sera consacrée à reprendre certains des théorèmes énoncés dans le papier "Some theorical properties of GANs" en vulgarisant certaines preuves et résultats voire en proposant de nouvelles démonstrations parfois plus simples.

## Présentation du problème et notations

Le principal objectif des GANs est d'approximer une distribution $p*$ en général beaucoup trop complioquées pour considérer qu'elle appartient à une famille paramétrique ou de l'approcher par le maximum de vraisemblance. Pour cela nous allons utiliser deux réseaux de neuronnes (pas réellement étudiés dans ce papier) et nous les allons mettre en compétition. Le premier sera le générateur qui aura pour objectif de générer un echantillon le plus proche possible de la vraie distribituion et le second appelé discriminateur aura pour objectif de savoir distinguer enter les vrais échantillons et les faux. Le discriminateur sera une fonction borélienne de l'espace des échantillons dans [0;1] et qui peut donc être considéré comme la probabilité d'être un vrai échantillon ou non. Nous allons considérer que toutes les familles de dsitributions considérées pour la génératioin sont dominées par une seule et même mesure $\mu$ . Ainsi notre GAN va simuler une variable aléatoire classique ( Gaussienne, exponentielle etc... en tant qu'entrée et va donc générer un échantillon grâce au reseau de neuronnes. Il faut savoir que bien la loi objectif $p*$ est généralement trop complexe pour considérer qu'elle appartient à une famille paramétrique, nous allons quand même prendre nos générateur et nos discriminateurs appartenant à une famille paramétrique où les indices appartiendront auront généralement de grandes dimensions.
   
Notons $Z$ la loi de la variable d'entrée et  $G_{\theta}$ avec $\theta\subseteq\ R^{p}$ la famille de générateurs de telle sorte que $G_{\theta}(Z)=p_{\theta}\mathrm{d}\mu$ avec $p_{\theta}\subseteq P$ et $P$ une famille de lois de paramètre $\theta$. De même, les discriminateurs $D_{\alpha}$ avec $\alpha \subseteq \mathrm{A} \subseteq R^{q}$.
Finalement, notons $X_1, \ldots, X_n$ nos données d'entraînement réelles et $Z_1, \ldots, Z_n$ celles simulées avec la loi connue pour entraîner nos deux réseaux de neuronnes. L'objectif pour le GAN est donc de trouver les paramètres $\alpha et \theta$ qui vont minimiser une certaine fonction coût et donc par conséquent s'interesser dans un premier temps à leur existence, puis à des questions de convergence si l'on agrandit l'espace des paramètres etc...
Dans notre papier, la fonction côut est la suivante: 
$$\inf_{\theta \in \mathit{\Theta}} \quad \sup_{D \in \mathit{D}}
\prod_{i=1}^{n} D(X_i) \times
\prod_{i=1}^{n} (1 - D \circ G_{\theta} (Z_i))^i
$$

$$
\hat{L}(\theta, D) \stackrel{\text { def }}{=} \sum_{i=1}^n \ln D\left(X_i\right)+\sum_{i=1}^n \ln \left(1-D \circ G_\theta\left(Z_i\right)\right)
$$

Dans cette fonction on peut percevoir deux parties la première: $\prod ^{n}_{i=1}D\left( x_{i}\right)$ 
où l'on veut maximiser cette valeur sachant que les $X_i$ sont les vraies données. De même, dans le deuxième terme, $\prod_{i=1}^n \ln \left(1-D \circ G_\theta\left(Z_i\right)\right)$ il faut donner la valeur la plus faible possible à $G_{\theta}\left(Z_i\right)$  tout en cherchant à minimiser cela sur $\theta$

Finalement, pour simplifier les calculs (comme on fait par exemple avec la vraisemblance) nous allons considérer le logartihme de cette fonction coût pour transformer les produits en sommes d'autant plus que nous possédons un nombre fini de $X_i$ et de $Z_i$ .

### Pourquoi cette fonction coût ?

Mis à part que cette fonction coût traduise parfaitement l'idée de la "compétition" entre le générateur et le discriminateur, elle a aussi été choisie pour sa forte ressemblance avec la divergence de Kullback-Leibler définie comme ci-dessous:

Si l'on considère $P$ et $Q$ deux mesures de probablités sur $E$,et $P$ presque sûrement continue par rapport à  $Q$, alors la divergence de Kullback-Leibler $Q$ envers $P$ est:
$$
D_{\mathrm{KL}}(P \| Q)=\int \ln \frac{\mathrm{d} P}{\mathrm{~d} Q} \mathrm{~d} P
$$
où $\frac{\mathrm{d} P}{\mathrm{~d} Q}$ est la dérivée selon Radon-Nikodym de $P$ par rapport à $Q$.

La divergence de KL est toujours positive. En voici la preuve dans le cas discret mais l'inégalité de convexité se transmet directement dans le cas continu.

$$
D_{\mathrm{KL}}(P \| Q)=\sum_i P(i) \log \frac{P(i)}{Q(i)}=-\sum_i P(i) \log \frac{Q(i)}{P(i)}
$$
Or le logarithme est strictement concave, d'où, en utilisant I' Inégalité de Jensen:
$$
\sum_i P(i) \log \frac{Q(i)}{P(i)} \leq \log \left(\sum_i P(i) \frac{Q(i)}{P(i)}\right)=\log \sum_i Q(i)=\log (1)=0
$$
Avec égalité ssi $\frac{Q(i)}{P(i)}$ est constant presque partout. (à cause de la stricte concavité) Dans ce cas-là, la constante ne peut qu'être égale à 1 puisque les deux fonctions $P$ et $Q$ sont des probabilités. Et on a bien que la divergence KL positive. Mais on remarque que celle-ci n'est pas symétrique. 

SI $p=\frac{\mathrm{d} P}{\mathrm{~d} \mu}$ et $q=\frac{\mathrm{d} Q}{\mathrm{~d} \mu}$ existent ce qui a été supposé au début en considérant que toutes les mesures de probabilités étaint dominées par $\mu$, alors la divergence de Kullback-Leibler est définie par:
$$
D_{\mathrm{KL}}(P \| Q)=\int p \ln \frac{p}{q} \mathrm{~d} \mu
$$
Mais on remarque que celle-ci n'est pas symétrique. On va donc la symétriser et considérer la divergence de Jensen-Shannon définie comme ci-dessous:
$$
D_{\mathrm{JS}}(P, Q)=\frac{1}{2} D_{\mathrm{KL}}\left(P \| \frac{P+Q}{2}\right)+\frac{1}{2} D_{\mathrm{KL}}\left(Q \| \frac{P+Q}{2}\right),
$$
qui vérifie$0 \leq D_{\mathrm{JS}}(P, Q) \leq \ln 2$. Le côté positif est immédiat comm somme de deux termes positifs (cf. démo juste au dessus). Montrons que c'est majoré dans le cas de deux mesures de probabilités, par $\ln 2$:

$D_{\mathrm{JS}}(P, Q)=\left( \dfrac{1}{2}\int p\ln \left( \dfrac{2p}{p+q}\right) +\dfrac{1}{2}\int q\ln \left( \dfrac{2q}{p+q}\right) \right)$=$\ln \left( 2\right) +\int p\ln \left( \dfrac{p}{p+q}\right) +\int q\ln \left( \dfrac{q}{p+q}\right)$
or $\ln\left(\frac{2p}{p+q}\right) \leq 0 $ et même pour $\ln\left(\frac{2q}{p+q}\right)$ ce qui permet de conclure
La racine carrée de la divergence de Jensen-Shannon est souvent considérée comme une distance. 
Revenons sur la fonction de coût et son lien avec la divergence de Jensen-Shannon. Pour l'étude théorique des propriétés des GANs nous allons considérer la version continue du logarithme de la fonction coût.
$$
L(\theta, D) \stackrel{\text { def }}{=} \int \ln (D) p^{\star} \mathrm{d} \mu+\int \ln (1-D) p_\theta \mathrm{d} \mu
$$

## Problèmes d'existence et d'unicité

Pour commencer nous allons considérer que l'ensemble des discriminateurs est infini qu'on notera $D_\infty$
$$
\begin{gathered}
\sup_{D \in D_{\infty}} L(\theta, D):=\ln (D) p^* d p+\int \ln (1-D) P_\theta d p . \\
0 \geqslant \sup_{D \in D_{\infty}} L(\theta, D) \geqslant(-\ln 2)\left\{\int p^{*} d p+\int p \theta d p\right\}=-2 \ln 2=-\ln 4 .
\end{gathered}
$$
$$
\begin{gathered}
\inf_{\theta \in \Theta} \sup_{D \in D_{\infty}} L(\theta, D) := \ln(D) p^* dp + \int \ln(1-D) P_\theta dp. \\
0 \geqslant \sup_{D \in D_{\infty}} L(\theta, D) \geqslant (-\ln 2) \left\{\int p^{+} dp + \int p \theta dp\right\} = -2 \ln 2 = -\ln 4.
\end{gathered}
$$

Et cela pour tout $\theta$

Nous allons nous intéresser qu'aux discriminateurs tels que $L(\theta, D)>-\infty$, qu'on dira $\theta$-admissibles pour limiter tous les problèmes d'intégrabilité. De même s'intéresser à $\mathscr{D}_{\infty}$ n'est que pour s'assurer des problèmes d'existence et créer le lien avec la divergence de Jensen-Shannon et dans les faits n'a aucun intérêt pratique. Si on Si on prend le sup de $L(\theta, D)$ sur $\mathscr{D}_{\infty}$ on a:
$$
\begin{aligned}
\sup _{D \in \mathscr{D}_{\infty}} L(\theta, D) & =\sup _{D \in \mathscr{D}_{\infty}} \int\left[\ln (D) p^{\star}+\ln (1-D) p_\theta\right] \mathrm{d} \mu \\
& \leq \int \sup _{D \in \mathscr{D}_{\infty}}\left[\ln (D) p^{\star}+\ln (1-D) p_\theta\right] \mathrm{d} \mu \\
& =L\left(\theta, D_\theta^{\star}\right)
\end{aligned}
$$
où
$$
D_\theta^{\star} \stackrel{\text { def }}{=} \frac{p^{\star}}{p^{\star}+p_\theta} .
$$
En effet, la première inégalité découle de l'inégalité triangulaire puis une étude de fonction de trouver $D_\theta^{\star}$
$\begin{aligned}
& \varphi: \quad d \mapsto \ln (d) p^*+b(1-d) p_{\theta}\quad \varphi^{\prime}(d)=\frac{p^*}{d}-\frac{p_{\theta}}{1-d}, \varphi^{\prime \prime}(d)=\frac{-p^*}{d^2}-\frac{p_{\theta}}{(1-d)^2} \leqslant 0 . \\
& d_*=\frac{p^*}{p_\theta+p^*} \text {. } \\
&
\end{aligned}
$


On peut en déduire que $\varphi$ est est concave et donc pour trouver le maximum il suffit de résoudre $\varphi$=0

Un calcul immédiat permet montrer que $L\left(\theta, D_\theta^{\star}\right)=2 D_{\mathrm{JS}}\left(p^{\star}, p_\theta\right)-\ln 4$ et de conclure que pour tout $\theta \in \Theta$,
$$
\sup _{D \in \mathscr{D}_{\infty}} L(\theta, D)=L\left(\theta, D_\theta^{\star}\right)=2 D_{\mathrm{JS}}\left(p^{\star}, p_\theta\right)-\ln 4
$$
En effet, en remplacant d* on a:


$
\begin{aligned}
& =2 \cdot \frac{1}{2} \int \ln \left(\frac{2 p^*}{p^*+p_\theta}\right) p^{\prime} d p+\int \ln \left(\frac{2 p_\theta}{p^*+p_\theta}\right) p_\theta d p .-2 \ln 2 . \\
& =2 D_{J S}\left(p^*, p_\theta\right)-\ln 4 .
\end{aligned}$

Et par conséquent minorer la fonction de coût suffit de minorer la divergence de Jensen-Shannon. 
De plus, il y a unicité du supremum sur le discriminateur mais celui-ci demande de connaître $p^*$ qui n'est généralement pas possible. Pour prouver l'unicité, le papier propose une démonstration très calculatoire et longue. En voici une plus courte:


Notons $\varphi(D)$ l'intérieur de l'intégrale

$$
\begin{aligned}
& \int [\underbrace{\varphi(\mathcal{D}_{\theta}^*) - \varphi(D)}] d\tilde{\mu} = 0\\
& \Rightarrow 0=\tilde{\mu}\left\{\varphi\left(D_\theta^*\right)-\varphi(D)>0\right\} \ge \tilde{\mu}\left(\left\{x / D(x) \neq D_\theta^*(x)\right\}\right) \Rightarrow D=D_\theta^* \quad \tilde{p}-a, s .
\end{aligned}
$$
Et donc $L\left(\theta, D_\theta^{\star}\right)=\sup _{D \in \mathscr{D}_{\infty}} L(\theta, D)=2 D_{\mathrm{JS}}\left(p^{\star}, p_\theta\right)-\ln 4, \quad \forall \theta \in \Theta$

Il est par conséquent intéressant de considérer au $\theta^*$ tel que:

$L\left(\theta^{\star}, D_{\theta^{\star}}^{\star}\right) \leq L\left(\theta, D_\theta^{\star}\right), \quad \forall \theta \in \Theta$

qui peut être interprété comme la meilleure façon d'approximer $p^*$ au sens de la divergence de Jensen-Shannon dans l'ensemble des $\Theta\$.
Mais il faut avant tout s'assurer de l'existence de ce paramètre.

Pour cela le théorème 2.2 de l'article nous assure cette existence sous certaines conditions:

Theorem 2.2. Supposons que le modèle $\left\{P_\theta\right\}_{\theta \in \Theta}$ est identifiable, convexe et compacte pour la métrique $\delta$.  Supposons, en outre, qu'il existe $0<m \leq M$ tel que $m \leq p^{\star} \leq M$ et pour tout $\theta \in \Theta, p_\theta \leq M$. Alors il existe un unique $\theta^{\star} \in \Theta$ tel que
$$
\left\{\theta^{\star}\right\}=\underset{\theta \in \Theta}{\arg \min } L\left(\theta, D_\theta^{\star}\right),
$$

Le théorème comme est énoncé ci-dessus présente en réalité une coquille car si une densité est minorée par une constante elle n'est tout simplement pas intégrable. Et de plus ceci n'est pas nécessaire dans la preuve:

### Existence

$L\left(\theta, D_\theta^*\right)=2 D_{J S}\left(p^{*}, p_\theta\right)-\ln 4 $

$\left(\left\{P_\theta / \theta \in \Theta\right\}, \delta \mid \stackrel{\varphi}{\rightarrow}(\mathbb{R}; \mid \mid) \right)
$

$P_\theta \mapsto \underbrace{2 D_{J s}\left(p^*, p_\theta\right)}_{\delta\left(p^{*}, p \theta\right)^2}-\operatorname{ln} 4 $

$\left|\psi\left(p_\theta\right)-\psi\left(\tilde{p}_\theta\right)\right|=2\left|\delta\left(p_p^*, p_\theta\right)-\delta\left(p^{+}, \tilde{p}_\theta\right)\right|^2 \leqslant 2\left|\delta\left(p_\theta, \tilde{p}_\theta\right)\right|^2 \Rightarrow \psi$ est continue
$\Rightarrow \Psi$ fonction continue sur un compact: $\left\{P_\theta / \theta \in \Theta\right\} \Rightarrow \exists \tilde{\theta} / \psi\left(P_{\tilde{\theta}}\right) .=\operatorname{Argmin}_{\tilde{\theta} \in\left\{P_\theta ; \theta \in \Theta(\Theta)\right.} \varphi\left(P_\theta\right)$

### Unicité

$\begin{aligned}
& \psi(p)=\int \ln \left(\frac{p^*}{p+p^*}\right) p^{+} d \mu+\int_\sigma \ln \left(\frac{2 p}{p+p^*}\right) p d p-\ln 4  \\
& =\int \ln \left(2p^*\right) p^* d\mu -ln 4 + \int[\underbrace{\left\{-\ln \left(p+p^*\right)\right\}\left(p+p^*\right)+p \ln 2 p}_{\psi_0(p)}] d\mu \\
&
\end{aligned}$


$\psi_0^{\prime}(p)=-1+\ln \left(p+p^*\right) \cdot+1+\ln (2p)$

$\varphi_0^{\prime \prime}(p)=-\frac{1}{p+p^*}+\frac{1}{p}=\frac{p^*}{p+p^*}>0$ et donc est strictement convexe

Par conséquent:

$\psi\left(\lambda p_\theta+(1-\lambda) \tilde{p_\theta}\right) \leq \lambda \psi\left(p_\theta\right)+(1-\lambda) \psi\left(\tilde{p}_\theta\right)$

$\text {De plus s'il y a égalité au lieu de} \leq \text { alors: } \int \lambda \cdot \psi_0(p_{\theta})+(1-\lambda) \cdot \psi(\tilde{p}_{\theta})-\left[\psi_0(\lambda \cdot p_{\theta}+(1-\lambda) \cdot \tilde{p}_{\theta})\right] d\mu = 0$

Par conséquent:

$0=\mu\left\{\lambda \varphi_0\left(p_{\theta}\right)+(1-\lambda) \psi\left(\tilde{p_0}\right)>\varphi_0\left(\lambda \cdot p_{\theta}+(1-\lambda) \tilde{p_0}\right)\right\} \geqslant \mu\left\{x \mid p_{\theta}(x) \neq \tilde{p}(x)\right\}$

Finalement si:

$$\psi\left(p_{\theta_1}\right)=\psi\left(p_{\theta_2}\right)=\min _{p_\theta \in\left\{p_\theta ; \theta \in \Theta\right.} \psi(p \theta) .
$$

Alors:

$\forall \lambda \in (0,1), \lambda p_{\theta_1} + (1-\lambda) p_{\theta_2} \in \left\{p_{\theta} ; \theta \in \Theta\right\}
$

Et: 
$\begin{aligned}
\psi\left(\lambda p_{\theta_1}+(1-\lambda) p_{\theta_2}\right) &= \psi(p_{\theta_1}) = \varphi(p_{\theta_2}) \\
&= \lambda \psi(\theta_1) + (1-\lambda) \varphi(\theta_2)
\end{aligned}$

Ce qui montre que:

$\mu\left\{x \mid P_{\theta_1}(x) \neq P_{\theta_2}(x)\right\}=0 \text { i.e: } P_{\theta_1}=P_{\theta_2} \quad \mu \text {-p.s}$

Ce qui prouve l'unicité.

L'existence et l'unicité peuvent être trouvées sous d'autres conditions moins réductrices mais nous ne les considérerons pas ici.

## Problèmes d'approximation

Supposons que $\theta^*$ existe. Puisque qu'en réalité nous de nous intéressons pas à tous les discriminateurs existants mais seulement à une certaine classe paramétrique, il est important de s'interroger sur la distance entre l'infémum sur $\theta$ dans l'ensemble considéré à l'infémum général. Nous allons le nommer $\overline{\theta}$.

De cette facon, $p_{\overline{\theta}}$ est le meilleur candidat pour approximer $p^*$, mais à quel point? peut-on quantifier la distance entre les deux? Est-ce que si $D$ augmente, $p_{\overline{\theta}}$ se rapproche de $p^*$ et à quelle vitesse? pour cela nous allons nous intéresser à la quantité suivante: $D_{\mathrm{JS}}\left(p^{\star}, p_{\bar{\theta}}\right)-D_{\mathrm{JS}}\left(p^{\star}, p_{\theta^{\star}}\right)$

Pour cela nous allons énoncer les conditions suivantes:

$\left(H_0\right)$ il existe une constante positive $\underline{t} \in(0,1 / 2]$ telle que
$$
\min \left(D_\theta^{\star}, 1-D_\theta^{\star}\right) \geq \underline{t}, \quad \forall \theta \in \Theta
$$
Rem: cela implique que $\theta \in \Theta$,
$$
\frac{\underline{t}}{1-\underline{t}} p^{\star} \leq p_\theta \leq \frac{1-\underline{t}}{\underline{t}} p^{\star} .
$$
et donc indépendamment du $\theta, p_\theta$ and $p^{\star}$ ont le même support.

$\left(H_{\varepsilon}\right)$ Il exitent $\varepsilon \in(0, \underline{t})$ et $D \in \mathscr{D}$, un $\bar{\theta}$-admissible discriminateur, tel que $\left\|D-D_{\bar{\theta}}^{\star}\right\|_{\infty} \leq \varepsilon$


Theorem 3.1. Sous $\left(H_0\right)$ et $\left(H_{\varepsilon}\right)$, il existe une constante $c$ (ne dépendant que de $\underline{t}$ ) telle que
$$
0 \leq D_{\mathrm{JS}}\left(p^{\star}, p_{\bar{\theta}}\right)-D_{\mathrm{JS}}\left(p^{\star}, p_{\theta^{\star}}\right) \leq c \varepsilon^2
$$
L'inégalité de gauche est immédiate, regardons celle de droite:

Notons:
$
\underbrace{|L(\bar{\theta}, D)-L\left(\bar{\theta}, D_{\bar{\theta}^*}\right)|}_{A(\bar{D})}
$

Puis le lemme suivant: $|\ln (r)-(r-1)| \leqslant|r-1| \ln r \mid$

Preuve:

Considérons la fonction suivante: $$\begin{aligned}
& \varphi(t)=\ln (r+t(r-1)) \\
& \varphi^{\prime}(t)=\frac{r-1}{1+t(r-1)}
\end{aligned}$$

Ainsi:

\begin{aligned}
& \varphi(1)-\left(\varphi(0)+\varphi^{\prime}(0)\right)=\int_0^1\left(\varphi^{\prime}(\mu)-\varphi^{\prime}(0)\right) d \mu . \\
& \ln (r)-(r-1) \\
& =\int_0^1(R-1)\left[\frac{1}{1+\mu(r-1)}-1\right] d u \text {. } \\
& =(r-1)^2 \int_0^1 \frac{-\mu}{1+\mu(r-1)} \, d\mu \\
&
\end{aligned}

Et le dénominateur appartient à [1;r] ou [r:1]

Donc:

$\begin{aligned}
|\ln r-(r-1)| \leqslant(r-1)^2 & \int_0^1 \frac{1}{1+\mu(r-1)} d u \\
& \frac{[\ln (\mu(r-1)+1)]_0^1}{r-1} \\
& =(r-1) \ln r
\end{aligned}$

Ce qui démontre le lemme.



Revenons à la preuve:


\begin{aligned}
& L(\bar{\theta}, D)-L\left(\bar{\theta}, D_{\bar{\theta}^*}\right)=\int\left[\ln \left(\frac{D}{D_{\bar{\theta}}^*}\right) p^*+\ln \left(\frac{1-D}{1-D_{\theta}}\right) p \bar{\theta} \cdot\right] d \mu . \\
& =\int\left[\left(\frac{D}{D_{\sigma}^{\alpha}}-1\right) p^t+\left(\frac{1-D}{1-D_{\theta}-1}\right) p \bar{\theta}\right] d \mu+R(D) \\
& =\int\left(D_0-\theta_{\theta}\right)\left[\frac{p^{\prime}}{D_{\theta}^2}-\frac{\rho \theta_*}{1-D_{\theta}^{\prime}}\right] d \psi+R(D) \text {. } \\
& {[\underbrace{p_{\bar{\theta}}+p^{+}-\left(p_{\bar{\theta}}+p^4\right)}_0]} \\
\end{aligned}

De plus,

$|R(D)| \leqslant \frac{2 \varepsilon^2}{\underline{t}(\underline{t}-\varepsilon)}$

Donc $L(\bar{\theta}, D_{\bar{\theta}}^*)=2 D_{J S}\left(p^*, p_{\bar{\theta}}\right)-\ln 4 \leqslant L(\bar{\theta}, D)+\frac{2 \varepsilon^2}{\underline{t}(\underline{t}-\varepsilon)} .$

$\leqslant \operatorname{sup}_{D \in D} L(\bar{\theta}, D)  +\frac{2 \varepsilon^2}{\underline{t}(\underline{t}-\varepsilon)}$

$\leq \operatorname{sup}_{D \in D} L\left(\theta^*, D\right)+\frac{2 \varepsilon^2}{\underline{t}(\underline{t}-\varepsilon)}$

$\leqslant \sup L\left(\theta^{+}, D\right)$ $D \in D_{\infty}+\frac{2 \varepsilon^2}{\underline{t}(\underline{t}-\varepsilon)}$

$\leq {\mathrm{DJ}}\left(p^{\prime}, p_{\theta_0}\right)-\ln 4+\frac{2 \varepsilon}{\underline{t}(\underline{t}-\varepsilon)}$

et pour finir: 
$
\begin{aligned}
& \theta_{J s}\left(p^{\prime}, p_{\bar{\theta}}\right) \leqslant D_{J S}\left(p^*, p_{\theta_\alpha}\right)+\underbrace{\left.\frac{\varepsilon^2}{t(\underline{t}-\varepsilon}\right)} \\
& \frac{\varepsilon^2}{\underline{\underline{t}}^2}\left(\frac{1}{1-\underline{\varepsilon}}\right) \\
& \sim 1+\frac{\varepsilon}{\underline{t}}+0\left(\frac{\varepsilon}{\underline{t}}\right) \\
& \frac{\varepsilon^2}{t^2}+\frac{\varepsilon^3}{t^3}+0\left(\frac{\varepsilon^3}{t^3}\right) \\
&
\end{aligned}
$


Finalement, la fin du papier s'intéresse aux propriétés assymptotiques de convergence vers la densité lorsque la taille des familles de paramètres augmente. Cela demande de savoir démontrer certains résultats comme l'inégalité de Dudley sur laquelle nous n'avons pas pu nous pencher.