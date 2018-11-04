from utils import *

##--------------------Load and Process data-------------------
# load ratings data 
#------------------
ratings = pd.read_csv('ml-100k/u.data',
                       sep='\t',
                       names=['userId','movieId','rating','time'])
# scale the movie ratings to [0,1]
ratings['rating'] = ratings['rating']/5
# convert timestamp to months since 1970 and then normalize it
ratings['time'] = ratings['time']/(3600*24*30)
ratings['time'] = (ratings['time']-ratings['time'].mean())\
                   /np.sqrt(ratings['time'].var())

# load users data & process it
#----------------------------
users  = pd.read_csv('ml-100k/u.user',sep='|',
                      names=['userId','age','gender','prof','zip'])
user_profs = pd.read_csv('ml-100k/u.occupation',names=['prof'])
# binarize gender
users['gender'] = users['gender'].apply(lambda x: 1 if x=='M' else 0)
# one-hot encode profession of users
users = pd.get_dummies(users,columns=['prof'])
# convert zip-codes to int values and normalize it
users['zip'] = users['zip'].apply(is_number)
users['zip'] = (users['zip']-users['zip'].mean())/np.sqrt(users['zip'].var())
# normalize age
users['age'] = (users['age']-users['age'].mean())/np.sqrt(users['age'].var())

# load movies data  & process it 
#-------------------------------
movie_genres = pd.read_csv('ml-100k/u.genre',sep='|',names=['genre','idx'])
movie_genres.drop(columns='idx',inplace=True)
movie_header = ['movieId', 'title', 'release_date',\
                'video_release_date', 'url']\
              +list(movie_genres['genre'])
movies = pd.read_csv('ml-100k/u2.item',sep='|',names=movie_header)

# drop some columns
movies.drop(columns=['title','video_release_date','url'],inplace=True)

# convert release-date to months passed since 1900 and then normalize it
movies['release_date'].fillna('01-Jan-1978',inplace=True)
movies['release_date'] = movies['release_date'].apply(months)
movies['release_date'] = (movies['release_date']-movies['release_date'].min())\
                         /np.sqrt(movies['release_date'].var())

##-----------Dataset----------------------
# main loop for training the neural network
max_epochs = 2 
batch_size = 1000
learning_rate = 0.002
split_ratio = 0.1 # train-test split ratio

# train-test split
train_ratings, test_ratings = train_test_split(ratings,test_size=split_ratio) 
train_data = RatingsDataset(train_ratings,users,movies)
test_data  = RatingsDataset(test_ratings,users,movies)

train_loader = DataLoader(train_data, batch_size=batch_size, 
                          shuffle=True, num_workers=4)
test_loader = DataLoader(test_data, batch_size=len(train_data))

##-----------Creating the Network & Traning----------------------
# create the neural network
# embedding_dim,n_users,n_movies,user_size,movie_size
nn_sizes = [200,50]
embedding_dim = 10
user_size = list(train_data[0]['user'].size())[0]
movie_size = list(train_data[0]['movie'].size())[0]
rec = NeuralRecommender(embedding_dim,len(users)+1,len(movies)+1,user_size,movie_size,nn_sizes)

##--------------Training----------------
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(rec.parameters(), lr=learning_rate)

print('training..')
loss_val = []
for it in range(max_epochs):
    for batch_it,batch in enumerate(train_loader):
        # Forward pass: Compute predicted y by passing X to the model
        yhat = rec.forward(batch['user'],batch['movie'])

        # Compute and print loss
        loss = criterion(yhat, batch['rating'])
        loss_val.append(loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  

    if np.mod(it,1)==0:
        print('epoch={:3d}, loss={}'.format(it+1,loss.item()))

# calculate test error
with torch.set_grad_enabled(False):
    for batch_it,batch in enumerate(test_loader):
        yhat = rec.forward(batch['user'],batch['movie'])
        loss = criterion(yhat, batch['rating'])

y_test = loss.item()
print('epochs={}, train error={}, test error={}'.format(max_epochs,loss_val[-1],y_test))

# display results
plt.plot(loss_val)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('training loss')

# save model parameters
#  torch.save(rec.state_dict(), 'model.pt')
# rec = NeuralRecommender(embedding_dim,len(users)+1,len(movies)+1,user_size,movie_size,nn_sizes)
#  rec.load_state_dict(torch.load('model.pt'))
#  model.forward(user,movie)
