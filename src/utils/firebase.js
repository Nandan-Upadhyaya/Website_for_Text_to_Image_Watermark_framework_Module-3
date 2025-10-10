import { initializeApp, getApps, getApp } from 'firebase/app';
import { getAuth, GoogleAuthProvider } from 'firebase/auth';

const firebaseConfig = {
	apiKey: process.env.REACT_APP_FIREBASE_API_KEY || 'AIzaSyCb1frTJCJnwPi83diomRCxsv2EAuJjapA',
	authDomain:
		process.env.REACT_APP_FIREBASE_AUTH_DOMAIN || `${process.env.REACT_APP_FIREBASE_PROJECT_ID || 'project-561719770763'}.firebaseapp.com`,
	projectId: process.env.REACT_APP_FIREBASE_PROJECT_ID || 'project-561719770763',
	storageBucket:
		process.env.REACT_APP_FIREBASE_STORAGE_BUCKET || `${process.env.REACT_APP_FIREBASE_PROJECT_ID || 'project-561719770763'}.appspot.com`,
	messagingSenderId: process.env.REACT_APP_FIREBASE_MESSAGING_SENDER_ID || '561719770763',
	appId: process.env.REACT_APP_FIREBASE_APP_ID || undefined,
};

const app = getApps().length ? getApp() : initializeApp(firebaseConfig);

const auth = getAuth(app);
const googleProvider = new GoogleAuthProvider();

googleProvider.setCustomParameters({
	prompt: 'select_account',
});

export { app, auth, googleProvider };
