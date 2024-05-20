import subprocess

def git_clone(repo_url, clone_dir):
    try:
        # Commande git clone
        result = subprocess.run(['git', 'clone', repo_url, clone_dir], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Afficher la sortie de la commande
        print(result.stdout.decode())
    except subprocess.CalledProcessError as e:
        # En cas d'erreur, afficher l'erreur
        print(f"Erreur lors de l'exécution de la commande git clone: {e.stderr.decode()}")
