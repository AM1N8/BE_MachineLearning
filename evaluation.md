Description

Pouvez-vous prédire qui utilise mon logiciel ? La société Infologic édite et intègre des solutions logicielles pour l’agroalimentaire depuis 1982. L'une des clés pour l'évolution et la maintenance de ces logiciels repose sur les tests continues effectués par des équipes internes. Ces testeurs utilisent divers profils pour tester différentes fonctionnalités. Pour chaque problème rencontré, ils doivent reporter leur utilisation du logiciel à partir de leur vrai profil, les obligeant à effectuer des manipulations parfois longues et rébarbatives.

C'est pourquoi nous proposons d'étudier s'il est possible d'identifier un utilisateur à partir de traces d'utilisation du logiciel (actions effectuées sur le logiciel) en construisant un modèle de classification utilisant des techniques de machine learning.

D'autre part, un tel modèle permettrait de diagnostiquer des tentatives d'intrusion logicielle, par exemple lorsqu'un pirate tente de se faire passer pour un utilisateur réel en se connectant avec son compte.

Vous disposez d'un vaste ensemble de données de traces logicielles, à partir desquelles vous devez créer des fonctionnalités et former un modèle. Une fois votre modèle prêt, appliquez-le aux données de test et soumettez votre résultat !

Bonne chance et bonne apprentissage !
Evaluation

La métrique d'évaluation pour ce concours est le F1 score moyen. Le F1 score, couramment utilisé dans la recherche d'informations, mesure la précision à l'aide des statistiques de précision p et de rappel r. La précision est le rapport entre les vrais positifs (tp) et tous les positifs prédits (tp + fp). Le rappel (recall) est le rapport entre les vrais positifs et tous les positifs réels (tp + fn). Le score F1 est donné par :
La métrique F1 pondère le rappel et la précision de manière égale, et un bon algorithme de recherche maximisera simultanément la précision et le rappel. Ainsi, une performance modérément bonne sur les deux sera préférée à une performance extrêmement bonne sur l'un et médiocre sur l'autre.